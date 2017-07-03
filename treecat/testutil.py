from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import shutil
import tempfile

import numpy as np

from treecat.config import DEFAULT_CONFIG

TINY_CONFIG = DEFAULT_CONFIG.copy()
TINY_CONFIG['learning_annealing_epochs'] = 2
TINY_CONFIG['model_num_clusters'] = 7

TINY_RAGGED_INDEX = np.array([0, 2, 4, 7, 10, 13], dtype=np.int32)
TINY_DATA = np.array(
    [
        #    |     |        |        |
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
