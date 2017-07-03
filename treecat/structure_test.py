from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.structure import OP_IN
from treecat.structure import OP_OUT
from treecat.structure import OP_ROOT
from treecat.structure import OP_UP
from treecat.structure import find_center_of_tree
from treecat.structure import make_complete_graph
from treecat.structure import make_propagation_schedule
from treecat.structure import make_tree
from treecat.structure import sample_tree
from treecat.testutil import numpy_seterr
from treecat.util import set_random_seed

numpy_seterr()

# https://oeis.org/A000272
NUM_SPANNING_TREES = [1, 1, 1, 3, 16, 125, 1296, 16807, 262144, 4782969]

EXAMPLE_TREES = [
    [],
    [(0, 1)],
    [(0, 1)],
    [(0, 1), (1, 2)],
    [(0, 1), (0, 2)],
    [(0, 2), (1, 2)],
    [(0, 1), (1, 2), (1, 3)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)],
]


@pytest.mark.parametrize('num_vertices,expected_grid', [
    (0, []),
    (1, []),
    (2, [[0], [0], [1]]),
    (3, [[0, 1, 2], [0, 0, 1], [1, 2, 2]]),
    (4, [[0, 1, 2, 3, 4, 5], [0, 0, 1, 0, 1, 2], [1, 2, 2, 3, 3, 3]]),
])
def test_make_complete_graph(num_vertices, expected_grid):
    V = num_vertices
    K = V * (V - 1) // 2
    expected_grid = np.array(expected_grid, dtype=np.int32).reshape([3, K])

    grid = make_complete_graph(V)
    np.testing.assert_array_equal(grid, expected_grid)


@pytest.mark.parametrize('edges,expected_grid', [
    ([], []),
    ([(0, 1)], [[0], [0], [1]]),
    ([(0, 1), (1, 2)], [[0, 1], [0, 1], [1, 2]]),
    ([(0, 1), (0, 2)], [[0, 1], [0, 0], [1, 2]]),
    ([(2, 1), (1, 0)], [[0, 1], [0, 1], [1, 2]]),
])
def test_make_tree(edges, expected_grid):
    E = len(edges)
    expected_grid = np.array(expected_grid, dtype=np.int32)
    expected_grid.shape = (3, E)

    grid = make_tree(edges)
    assert grid.shape == expected_grid.shape
    assert (grid == expected_grid).all()


@pytest.mark.parametrize('edges', EXAMPLE_TREES)
def test_make_tree_runs(edges):
    E = len(edges)
    V = E + 1
    grid = make_tree(edges)
    assert grid.shape == (3, len(edges))
    if edges:
        assert set(grid[0, :]) == set(range(E))
        assert set(grid[1, :]) | set(grid[2, :]) == set(range(V))


@pytest.mark.parametrize('expected_vertex,edges', [
    (0, []),
    (0, [(0, 1)]),
    (0, [(0, 1), (0, 2)]),
    (1, [(0, 1), (1, 2)]),
    (2, [(0, 2), (1, 2)]),
    (1, [(0, 1), (1, 2), (2, 3)]),
    (1, [(0, 1), (1, 2), (1, 3)]),
    (2, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    (2, [(0, 2), (1, 2), (2, 3), (2, 4)]),
])
def test_find_center_of_tree(expected_vertex, edges):
    grid = make_tree(edges)

    v = find_center_of_tree(grid)
    assert v == expected_vertex


EXAMPLE_ROOTED_TREES = [(edges, root)
                        for edges in EXAMPLE_TREES
                        for root in [None] + list(range(1 + len(edges)))]


@pytest.mark.parametrize('edges,root', EXAMPLE_ROOTED_TREES)
def test_make_propagation_schedule(edges, root):
    E = len(edges)
    V = E + 1
    grid = make_tree(edges)
    neighbors = {v: set() for v in range(V)}
    for e, v1, v2 in grid.T:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)

    # Generate a schedule.
    schedule = make_propagation_schedule(grid, root)
    assert schedule.shape == (V + E + 1 + E, 4)
    assert schedule.dtype == np.int16

    # Check topology.
    if root is not None:
        assert schedule[V + E][0] == OP_ROOT
        assert schedule[V + E][1] == root
    assert set(row[1] for row in schedule) == set(range(V)), 'bad vertex set'
    for op, v, v2, e in schedule:
        if op == OP_ROOT:
            if root is not None:
                assert v == root
        elif op != OP_UP:
            assert v != v2
            assert v2 in neighbors[v]
            assert grid[1, e] == min(v, v2)
            assert grid[2, e] == max(v, v2)

    # Check inward ordering.
    state = np.zeros(V, np.int8)
    for op, v, v2, e in schedule:
        if op == OP_UP:
            assert state[v] == 0
            state[v] = 1
        elif op == OP_IN:
            assert state[v] == 1
            assert state[v2] == 1
        elif op == OP_ROOT:
            assert state[v] == 1
            state[v] = 2
        elif op == OP_OUT:
            assert state[v] == 1
            assert state[v2] == 2
            state[v] = 2
    assert np.all(state == 2)


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4])
def test_sample_tree_gof(num_edges):
    set_random_seed(0)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = np.random.random([K])
    edge_probs = np.exp(edge_logits)
    edge_probs_dict = {(v1, v2): edge_probs[k] for k, v1, v2 in grid.T}

    # Generate many samples via MCMC.
    num_samples = 2000
    counts = defaultdict(lambda: 0)
    edges = [(v, v + 1) for v in range(V - 1)]
    for _ in range(num_samples):
        edges = sample_tree(grid, edge_logits, edges)
        counts[tuple(edges)] += 1
    assert len(counts) == NUM_SPANNING_TREES[V]

    # Check accuracy using Pearson's chi-squared test.
    keys = counts.keys()
    counts = np.array([counts[key] for key in keys])
    probs = np.array(
        [np.prod([edge_probs_dict[edge] for edge in key]) for key in keys])
    probs /= probs.sum()
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof
