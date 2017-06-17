from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os
from copy import deepcopy

import numpy as np

from treecat.testutil import assert_equal
from treecat.testutil import tempdir
from treecat.training import DEFAULT_CONFIG
from treecat.training import Model

TINY_CONFIG = deepcopy(DEFAULT_CONFIG)
TINY_CONFIG['annealing']['epochs'] = 2

TINY_DATA = np.array(
    [
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 1],
        [1, 0, 2, 2, 2],
        [1, 0, 0, 0, 1],
    ],
    dtype=np.int32)

TINY_MASK = np.array(
    [
        [1, 1, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1],
    ],
    dtype=np.int32)


def test_model_init_runs():
    Model(TINY_DATA, TINY_MASK)


def test_model_fit_runs():
    model = Model(TINY_DATA, TINY_MASK, TINY_CONFIG)
    model.fit()


def test_model_save_load():
    model = Model(TINY_DATA, TINY_MASK, TINY_CONFIG)
    model.fit()
    with tempdir() as dirname:
        filename = os.path.join(dirname, 'model.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        with open(filename, 'rb') as f:
            model2 = pickle.load(f)

    assert_equal(model2._data, model._data)
    assert_equal(model2._mask, model._mask)
    assert_equal(model2._config, model._config)
    assert_equal(model2._seed, model._seed)
    assert_equal(model2._assignments, model._assignments)
    assert_equal(model2._variables, model._variables)
