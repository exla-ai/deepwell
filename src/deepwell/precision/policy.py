from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set


MXFP8 = "mxfp8"
BF16 = "bf16"
FP4 = "fp4"  # via NVFP4 path later, guarded


@dataclass
class PrecisionPolicy:
    default_dtype: str = MXFP8
    bf16_layers: Set[str] = field(default_factory=lambda: {"embeddings", "first_linear", "last_linear", "logits", "norms"})
    fp4_layers: Set[str] = field(default_factory=set)
    transpose_points: Set[str] = field(default_factory=set)
    dual_buffer_scales: bool = True
    guard_auto_fallback: bool = True

    def dtype_for(self, layer_name: str) -> str:
        if layer_name in self.bf16_layers:
            return BF16
        if layer_name in self.fp4_layers:
            return FP4
        return self.default_dtype

    def set_layer_dtype(self, layer_name: str, dtype: str) -> None:
        if dtype == BF16:
            self.bf16_layers.add(layer_name)
            self.fp4_layers.discard(layer_name)
        elif dtype == FP4:
            self.fp4_layers.add(layer_name)
            self.bf16_layers.discard(layer_name)
        elif dtype == MXFP8:
            self.fp4_layers.discard(layer_name)
            self.bf16_layers.discard(layer_name)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def mark_transpose_point(self, tensor_name: str) -> None:
        self.transpose_points.add(tensor_name)

    def is_transpose_requantized(self, tensor_name: str) -> bool:
        return tensor_name in self.transpose_points


