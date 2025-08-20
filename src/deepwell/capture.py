"""Model capture module using PyTorch FX for graph extraction."""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.fx as fx
from torch.fx import GraphModule, Node
import torch.nn as nn

from .ir import IR, Op, TensorTy, TensorId, Kind


class ModelTracer:
    """Traces PyTorch models to create an intermediate representation."""
    
    def __init__(self):
        self.tensor_counter = 0
        self.op_counter = 0
        self.tensors: Dict[TensorId, TensorTy] = {}
        self.params: Dict[TensorId, TensorTy] = {}
        self.ops: List[Op] = []
        self.node_to_tensor_id: Dict[Node, TensorId] = {}
        
    def _get_next_tensor_id(self, prefix: str = "t") -> TensorId:
        """Generate unique tensor ID."""
        tid = f"{prefix}_{self.tensor_counter}"
        self.tensor_counter += 1
        return tid
    
    def _get_next_op_id(self, kind: str = "op") -> str:
        """Generate unique operation ID."""
        op_id = f"{kind}_{self.op_counter}"
        self.op_counter += 1
        return op_id
    
    def _infer_dtype(self, tensor: Union[torch.Tensor, torch.dtype, None]) -> str:
        """Infer data type from tensor or dtype."""
        if tensor is None:
            return "f32"
        
        if isinstance(tensor, torch.Tensor):
            dtype = tensor.dtype
        else:
            dtype = tensor
            
        dtype_map = {
            torch.float32: "f32",
            torch.float16: "f16",
            torch.bfloat16: "bf16",
            torch.float64: "f64",
            torch.int32: "i32",
            torch.int64: "i64",
            torch.int8: "i8",
            torch.uint8: "u8",
            torch.bool: "bool",
        }
        return dtype_map.get(dtype, "f32")
    
    def _infer_shape(self, node: Node) -> Tuple[Union[int, str], ...]:
        """Infer shape from FX node."""
        # Try to get shape from meta information
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            if hasattr(meta, 'shape'):
                return tuple(meta.shape)
        
        # Try to get from example value
        if hasattr(node, 'meta') and 'example_value' in node.meta:
            val = node.meta['example_value']
            if isinstance(val, torch.Tensor):
                return tuple(val.shape)
        
        # Default shape
        return ('?',)  # Unknown shape
    
    def _classify_op(self, node: Node) -> Kind:
        """Classify operation type based on FX node."""
        if node.op == 'call_module':
            module_name = node.target
            if 'linear' in str(module_name).lower():
                return 'linear'
            elif 'attention' in str(module_name).lower():
                return 'attention'
            elif any(norm in str(module_name).lower() for norm in ['norm', 'layernorm', 'batchnorm']):
                return 'norm'
            elif 'embed' in str(module_name).lower():
                return 'embed'
            elif 'mlp' in str(module_name).lower() or 'feedforward' in str(module_name).lower():
                return 'mlp'
        elif node.op == 'call_function':
            func_name = str(node.target)
            if 'linear' in func_name:
                return 'linear'
            elif 'matmul' in func_name or 'mm' in func_name:
                return 'linear'
            elif 'softmax' in func_name or 'attention' in func_name:
                return 'attention'
        
        return 'other'
    
    def _extract_attrs(self, node: Node, module: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Extract attributes from node and associated module."""
        attrs = {}
        
        # Add source information
        if node.op == 'call_module' and node.target:
            attrs['source'] = str(node.target)
        
        # Extract module-specific attributes
        if module is not None:
            if isinstance(module, nn.Linear):
                attrs['in_features'] = module.in_features
                attrs['out_features'] = module.out_features
                attrs['has_bias'] = module.bias is not None
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if hasattr(module, 'normalized_shape'):
                    attrs['normalized_shape'] = module.normalized_shape
                attrs['eps'] = module.eps if hasattr(module, 'eps') else 1e-5
            elif isinstance(module, nn.Embedding):
                attrs['num_embeddings'] = module.num_embeddings
                attrs['embedding_dim'] = module.embedding_dim
        
        # Add shape information if available
        if hasattr(node, 'meta'):
            if 'tensor_meta' in node.meta:
                meta = node.meta['tensor_meta']
                if hasattr(meta, 'shape'):
                    attrs['output_shape'] = list(meta.shape)
        
        return attrs
    
    def trace_module(self, model: nn.Module, example_inputs: Optional[torch.Tensor] = None) -> IR:
        """
        Trace a PyTorch module and convert to IR.
        
        Args:
            model: PyTorch module to trace
            example_inputs: Example input tensor for tracing
            
        Returns:
            IR representation of the model
        """
        # Create example input if not provided
        if example_inputs is None:
            # Try to infer from first layer
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    example_inputs = torch.randn(1, module.in_features)
                    break
                elif isinstance(module, nn.Embedding):
                    example_inputs = torch.randint(0, module.num_embeddings, (1, 128))
                    break
            
            if example_inputs is None:
                # Default to common transformer input
                example_inputs = torch.randn(1, 512, 768)  # [batch, seq_len, hidden_dim]
        
        # Trace the model
        try:
            traced = fx.symbolic_trace(model)
        except Exception as e:
            # Fallback to torch.jit.trace if FX fails
            print(f"FX tracing failed: {e}. Attempting manual graph construction...")
            return self._manual_trace(model, example_inputs)
        
        # Process the graph
        graph = traced.graph
        modules = dict(traced.named_modules())
        
        # First pass: collect parameters
        for name, param in model.named_parameters():
            param_id = f"param_{name.replace('.', '_')}"
            shape = tuple(param.shape)
            dtype = self._infer_dtype(param)
            
            self.params[param_id] = TensorTy(
                shape=shape,
                dtype=dtype,
                layout="contig"
            )
        
        # Second pass: process operations
        for node in graph.nodes:
            if node.op == 'placeholder':
                # Input tensors
                tensor_id = self._get_next_tensor_id("input")
                shape = self._infer_shape(node)
                dtype = self._infer_dtype(None)
                
                self.tensors[tensor_id] = TensorTy(
                    shape=shape,
                    dtype=dtype,
                    layout="contig"
                )
                self.node_to_tensor_id[node] = tensor_id
                
            elif node.op in ['call_module', 'call_function', 'call_method']:
                # Operations
                kind = self._classify_op(node)
                op_id = self._get_next_op_id(kind)
                
                # Get input tensor IDs
                input_ids = []
                for arg in node.args:
                    if isinstance(arg, Node) and arg in self.node_to_tensor_id:
                        input_ids.append(self.node_to_tensor_id[arg])
                    elif isinstance(arg, (list, tuple)):
                        for a in arg:
                            if isinstance(a, Node) and a in self.node_to_tensor_id:
                                input_ids.append(self.node_to_tensor_id[a])
                
                # Create output tensor
                output_id = self._get_next_tensor_id()
                shape = self._infer_shape(node)
                dtype = self._infer_dtype(None)
                
                self.tensors[output_id] = TensorTy(
                    shape=shape,
                    dtype=dtype,
                    layout="contig"
                )
                self.node_to_tensor_id[node] = output_id
                
                # Get module if this is a module call
                module = None
                if node.op == 'call_module' and node.target in modules:
                    module = modules[node.target]
                
                # Extract attributes
                attrs = self._extract_attrs(node, module)
                
                # Create operation
                op = Op(
                    id=op_id,
                    kind=kind,
                    inputs=input_ids,
                    outputs=[output_id],
                    attrs=attrs
                )
                self.ops.append(op)
                
            elif node.op == 'output':
                # Output node - no operation needed
                pass
        
        return IR(
            tensors=self.tensors,
            params=self.params,
            ops=self.ops
        )
    
    def _manual_trace(self, model: nn.Module, example_inputs: torch.Tensor) -> IR:
        """
        Manually trace model by iterating through modules.
        Fallback when FX tracing fails.
        """
        # Collect all modules in order
        modules_list = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                modules_list.append((name, module))
        
        # Create input tensor
        input_id = self._get_next_tensor_id("input")
        input_shape = tuple(example_inputs.shape)
        input_dtype = self._infer_dtype(example_inputs)
        
        self.tensors[input_id] = TensorTy(
            shape=input_shape,
            dtype=input_dtype,
            layout="contig"
        )
        
        last_output_id = input_id
        
        # Process each module
        for name, module in modules_list:
            # Classify module type
            if isinstance(module, nn.Linear):
                kind = 'linear'
            elif isinstance(module, nn.MultiheadAttention):
                kind = 'attention'
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                kind = 'norm'
            elif isinstance(module, nn.Embedding):
                kind = 'embed'
            else:
                kind = 'other'
            
            # Skip non-computation modules
            if isinstance(module, (nn.Dropout, nn.Identity)):
                continue
            
            # Create operation
            op_id = self._get_next_op_id(kind)
            output_id = self._get_next_tensor_id()
            
            # Infer output shape (simplified)
            if isinstance(module, nn.Linear):
                output_shape = input_shape[:-1] + (module.out_features,)
            elif isinstance(module, nn.Embedding):
                output_shape = (input_shape[0], input_shape[1], module.embedding_dim)
            else:
                output_shape = input_shape
            
            self.tensors[output_id] = TensorTy(
                shape=output_shape,
                dtype=input_dtype,
                layout="contig"
            )
            
            # Extract attributes
            attrs = {'source': name}
            if isinstance(module, nn.Linear):
                attrs.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'has_bias': module.bias is not None
                })
            
            op = Op(
                id=op_id,
                kind=kind,
                inputs=[last_output_id],
                outputs=[output_id],
                attrs=attrs
            )
            self.ops.append(op)
            
            last_output_id = output_id
            input_shape = output_shape
        
        # Add parameters
        for name, param in model.named_parameters():
            param_id = f"param_{name.replace('.', '_')}"
            self.params[param_id] = TensorTy(
                shape=tuple(param.shape),
                dtype=self._infer_dtype(param),
                layout="contig"
            )
        
        return IR(
            tensors=self.tensors,
            params=self.params,
            ops=self.ops
        )


def capture(model: nn.Module, example_inputs: Optional[torch.Tensor] = None) -> IR:
    """
    Capture a PyTorch model as an intermediate representation.
    
    Args:
        model: PyTorch model to capture
        example_inputs: Optional example input for tracing
        
    Returns:
        IR representation of the model
    """
    tracer = ModelTracer()
    return tracer.trace_module(model, example_inputs)


def print_ir(ir: IR) -> None:
    """Pretty print the IR for debugging."""
    print("=" * 60)
    print("Model IR")
    print("=" * 60)
    
    print(f"\nParameters: {len(ir.params)}")
    for param_id, param_ty in list(ir.params.items())[:5]:
        print(f"  {param_id}: shape={param_ty.shape}, dtype={param_ty.dtype}")
    if len(ir.params) > 5:
        print(f"  ... and {len(ir.params) - 5} more")
    
    print(f"\nTensors: {len(ir.tensors)}")
    for tensor_id, tensor_ty in list(ir.tensors.items())[:5]:
        print(f"  {tensor_id}: shape={tensor_ty.shape}, dtype={tensor_ty.dtype}")
    if len(ir.tensors) > 5:
        print(f"  ... and {len(ir.tensors) - 5} more")
    
    print(f"\nOperations: {len(ir.ops)}")
    for op in ir.ops[:10]:
        print(f"  [{op.id}] {op.kind}:")
        print(f"    inputs: {op.inputs}")
        print(f"    outputs: {op.outputs}")
        if 'source' in op.attrs:
            print(f"    source: {op.attrs['source']}")
    if len(ir.ops) > 10:
        print(f"  ... and {len(ir.ops) - 10} more")
    
    print("=" * 60)
