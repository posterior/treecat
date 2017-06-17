import contextlib

import numpy as np
import pytest


def assert_equal(x, y):
    assert type(x) == type(y), (x, y)
    if isinstance(x, dict):
        assert x.keys() == y.keys(), (x, y)
        for key in x.keys():
            assert_equal(x[key], y[key])
    elif isinstance(x, np.ndarray):
        assert np.all(x == y), (x, y)
    else:
        assert x == y, (x, y)


@contextlib.contextmanager
def xfail_if_not_implemented():
    try:
        yield
    except NotImplementedError as e:
        pytest.xfail(reason=str(e))
