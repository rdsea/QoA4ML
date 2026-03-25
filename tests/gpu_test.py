import pytest


@pytest.mark.requires_gpu
def test_find_igpu():
    from qoa4ml.utils.jetson_utils import find_igpu

    result = find_igpu()
    assert isinstance(result, (dict, type(None)))
