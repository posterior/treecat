from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from treecat.engine import Model, make_complete_graph


@pytest.mark.parametrize('num_vertices,expected_grid', [
    (0, []),
    (1, []),
    (2, [[0], [0], [1]]),
    (3, [[0, 1, 2], [0, 0, 1], [1, 2, 2]]),
])
def test_make_complete_graph(num_vertices, expected_grid):
    num_edges = num_vertices * (num_vertices - 1) // 2
    expected_grid = np.array(expected_grid, dtype=np.int32)
    expected_grid.shape = (3, num_edges)
    V, E, grid = make_complete_graph(num_vertices)
    assert V == num_vertices
    assert E == num_edges
    assert grid.shape == expected_grid.shape
    assert (grid == expected_grid).all()


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


def test_create_model():
    Model(TINY_DATA, TINY_MASK)


@pytest.mark.xfail
def test_create_graph():
    model = Model(TINY_DATA, TINY_MASK)
    model.update_session()
