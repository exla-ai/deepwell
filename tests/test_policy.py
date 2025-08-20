from deepwell import PrecisionPolicy


def test_policy_defaults_and_overrides():
    p = PrecisionPolicy()
    # Defaults
    assert p.dtype_for("mlp") == "mxfp8"
    assert p.dtype_for("norms") == "bf16"
    # Override to FP4 (guarded later)
    p.set_layer_dtype("mlp", "fp4")
    assert p.dtype_for("mlp") == "fp4"
    # Transpose-point tracking
    p.mark_transpose_point("attn_qk")
    assert p.is_transpose_requantized("attn_qk") is True


