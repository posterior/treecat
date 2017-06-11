from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from treecat.engine import Model

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


@pytest.mark.xfail
def test_create_model():
    Model(TINY_DATA, TINY_MASK)
