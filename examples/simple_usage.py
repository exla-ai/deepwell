#!/usr/bin/env python3
"""
Simple usage example for Deepwell library
Shows how to use the high-performance attention module
"""

import torch
from deepwell.kernels.blackwell_production import (
    BlackwellFlashAttention,
    BlackwellConfig,
    DWSelfAttention
)

def example_basic_attention():
    """Basic usage of BlackwellFlashAttention"""
    print("Example 1: Basic Flash Attention")
    print("-" * 40)
    
    # Create configuration
    config = BlackwellConfig()
    
    # Initialize Flash Attention
    flash_attn = BlackwellFlashAttention(config)
    
    # Create sample tensors (B=batch, H=heads, S=sequence, D=dimension)
    batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.bfloat16)
    
    # Run attention
    output = flash_attn.forward(q, k, v, causal=True)
    
    print(f"Input shape:  {q.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_self_attention_module():
    """Using DWSelfAttention as a drop-in replacement"""
    print("Example 2: Self-Attention Module")
    print("-" * 40)
    
    # Configuration
    hidden_dim = 512
    num_heads = 8
    config = BlackwellConfig()
    
    # Create self-attention module
    attention = DWSelfAttention(hidden_dim, num_heads, config)
    
    # Create input tensor [batch, sequence, hidden_dim]
    batch_size, seq_len = 4, 256
    x = torch.randn(batch_size, seq_len, hidden_dim,
                    device='cuda', dtype=torch.bfloat16)
    
    # Apply attention
    output = attention(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_transformer_layer():
    """Building a simple transformer layer"""
    print("Example 3: Transformer Layer")
    print("-" * 40)
    
    import torch.nn as nn
    
    class TransformerLayer(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super().__init__()
            config = BlackwellConfig()
            self.attention = DWSelfAttention(hidden_dim, num_heads, config)
            self.norm1 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
            self.norm2 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim, dtype=torch.bfloat16),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, hidden_dim, dtype=torch.bfloat16)
            )
        
        def forward(self, x):
            # Self-attention with residual
            attn_out = self.attention(self.norm1(x))
            x = x + attn_out
            
            # FFN with residual
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out
            
            return x
    
    # Create and use the layer
    layer = TransformerLayer(512, 8).cuda()
    x = torch.randn(2, 128, 512, device='cuda', dtype=torch.bfloat16)
    output = layer(x)
    
    print(f"Transformer layer created")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def main():
    """Run all examples"""
    import os
    
    # Set environment variables for CUTLASS bridge
    os.environ['DW_ENABLE_FMHA_BRIDGE'] = '1'
    
    # Check if custom bridge path exists
    bridge_path = '/root/deepwell/csrc/fmha_bridge_min/build/libdw_fmha_bridge_min.so'
    if os.path.exists(bridge_path):
        os.environ['DW_FMHA_BRIDGE_PATH'] = bridge_path
    
    print("=" * 50)
    print("Deepwell Usage Examples")
    print("=" * 50)
    print()
    
    # Run examples
    example_basic_attention()
    example_self_attention_module()
    example_transformer_layer()
    
    print("All examples completed successfully!")


if __name__ == '__main__':
    main()
