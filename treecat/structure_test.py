from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from treecat.structure import make_complete_graph


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
