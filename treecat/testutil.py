from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import shutil
import tempfile

import numpy as np

from treecat.config import make_config

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')

TINY_CONFIG = make_config(
    learning_init_epochs=2,  #
    model_num_clusters=7,  #
    model_ensemble_size=3)

TINY_RAGGED_INDEX = np.array([0, 2, 4, 7, 10, 13], dtype=np.int32)
TINY_DATA = np.array(
    [
        # f1 | f2  |   f3   |   f4   |   f5   |
        # ---+-----+--------+--------+--------+
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ],
    dtype=np.int8)


def numpy_seterr():
    np.seterr(divide='raise', invalid='raise')


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
