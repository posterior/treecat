from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import shutil
import tempfile

import numpy as np
import pytest


@contextlib.contextmanager
def tempdir():
    dirname = tempfile.mkdtemp()
    try:
        yield dirname
    finally:
        shutil.rmtree(dirname)


def assert_equal(x, y):
    assert type(x) == type(y), (x, y)
    if isinstance(x, dict):
        assert x.keys() == y.keys(), (x, y)
        for key in x.keys():
            assert_equal(x[key], y[key])
    elif isinstance(x, np.ndarray):
        np.testing.assert_array_equal(x, y)
    else:
        assert x == y, (x, y)


@contextlib.contextmanager
def xfail_if_not_implemented():
    try:
        yield
    except NotImplementedError as e:
        pytest.xfail(reason=str(e))
