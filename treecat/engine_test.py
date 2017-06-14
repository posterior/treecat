from __future__ import absolute_import, division, print_function

from copy import deepcopy

import numpy as np

from treecat.engine import DEFAULT_CONFIG, Model

TINY_CONFIG = deepcopy(DEFAULT_CONFIG)
TINY_CONFIG['annealing']['epochs'] = 4

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


def test_model_init():
    Model(TINY_DATA, TINY_MASK)


def test_model_fit():
    model = Model(TINY_DATA, TINY_MASK, TINY_CONFIG)
    model.fit()
