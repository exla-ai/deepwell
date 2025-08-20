import importlib


def test_imports_from_readme_usage():
    dw = importlib.import_module("deepwell")
    assert hasattr(dw, "probe")
    assert hasattr(dw, "capture")
    assert hasattr(dw, "PrecisionPolicy")


