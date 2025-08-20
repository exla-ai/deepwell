import pytest

from deepwell import capture


def test_capture_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        capture(object())


