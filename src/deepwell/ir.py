from typing import Tuple, Dict, List, Union, Literal
from dataclasses import dataclass

TensorId = str
DType = str
Layout = Literal["row","col","contig"]
Kind = Literal["linear","attention","mlp","norm","embed","other"]
AttrV = Union[int, float, str, bool]

@dataclass(frozen=True)
class TensorTy:
    shape: Tuple[Union[int,str], ...]
    dtype: DType
    layout: Layout

@dataclass(frozen=True)
class Op:
    id: str
    kind: Kind
    inputs: List[TensorId]
    outputs: List[TensorId]
    attrs: Dict[str, AttrV]  # include {"source": "layer.path"} when capturing

@dataclass(frozen=True)
class IR:
    tensors: Dict[TensorId, TensorTy]
    params:  Dict[TensorId, TensorTy]
    ops:     List[Op]
