from __future__ import absolute_import, division, print_function

from collections import defaultdict

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.structure import (find_center_of_tree, find_connected_component,
                               make_complete_graph, make_propagation_schedule,
                               make_tree, sample_tree)

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


@pytest.mark.parametrize('edges,expected_grid', [
    ([], []),
    ([(0, 1)], [[0], [0], [1]]),
    ([(0, 1), (1, 2)], [[0, 1], [0, 1], [1, 2]]),
    ([(0, 1), (0, 2)], [[0, 1], [0, 0], [1, 2]]),
    ([(2, 1), (1, 0)], [[0, 1], [0, 1], [1, 2]]),
])
def test_make_tree(edges, expected_grid):
    num_edges = len(edges)
    num_vertices = 1 + num_edges
    expected_grid = np.array(expected_grid, dtype=np.int32)
    expected_grid.shape = (3, num_edges)

    V, E, grid = make_tree(edges)
    assert V == num_vertices
    assert E == num_edges
    assert grid.shape == expected_grid.shape
    assert (grid == expected_grid).all()


@pytest.mark.parametrize('edges', EXAMPLE_TREES)
def test_make_tree_runs(edges):
    V, E, grid = make_tree(edges)
    assert grid.shape == (3, E)
    if E > 0:
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
    V, E, grid = make_tree(edges)

    v = find_center_of_tree(grid)
    assert v == expected_vertex


EXAMPLE_ROOTED_TREES = [(edges, root)
                        for edges in EXAMPLE_TREES
                        for root in [None] + list(range(1 + len(edges)))]


@pytest.mark.parametrize('edges,root', EXAMPLE_ROOTED_TREES)
def test_make_propagation_schedule(edges, root):
    V, E, grid = make_tree(edges)
    neighbors = {v: set() for v in range(V)}
    for e, v1, v2 in grid.T:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)

    schedule = make_propagation_schedule(grid, root)
    if root is not None:
        assert schedule[0][0] == root
    assert schedule[0][1] is None, 'root has a parent'
    assert set(task[0] for task in schedule) == set(range(V)), 'bad vertex set'
    assert all(task[1] is not None for task in schedule[1:]), 'missing parent'
    for v, parent, children in schedule:
        actual_neighbors = set(children)
        if parent is not None:
            actual_neighbors.add(parent)
        assert actual_neighbors == neighbors[v]


@pytest.mark.parametrize('V,edges,v,expected_component', [
    (4, [(0, 3), (1, 2)], 0, set([0, 3])),
    (4, [(0, 3), (1, 2)], 1, set([1, 2])),
    (4, [(0, 3), (1, 2)], 2, set([1, 2])),
    (4, [(0, 3), (1, 2)], 3, set([0, 3])),
    (4, [(1, 2), (2, 3)], 0, set([0])),
    (4, [(1, 2), (2, 3)], 1, set([1, 2, 3])),
    (4, [(1, 2), (2, 3)], 2, set([1, 2, 3])),
    (4, [(1, 2), (2, 3)], 3, set([1, 2, 3])),
])
def test_find_connected_component(V, edges, v, expected_component):
    neighbors = [set() for _ in range(V)]
    for v1, v2 in edges:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)

    component = find_connected_component(neighbors, v)
    assert component == expected_component


# https://oeis.org/A000272
NUM_SPANNING_TREES = [1, 1, 1, 3, 16, 125, 1296, 16807, 262144, 4782969]


@pytest.mark.parametrize('edges', [
    [(0, 1)],
    pytest.mark.xfail([(0, 1), (1, 2)]),
    pytest.mark.xfail([(0, 1), (1, 2), (2, 3)]),
    pytest.mark.xfail([(0, 1), (1, 2), (2, 3), (3, 4)]),
])
def test_sample_tree(edges):
    np.random.seed(0)
    E = len(edges)
    V = 1 + E
    V, E, grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_prob = np.exp(-np.random.random([K]))
    edge_prob_dict = {(v1, v2): edge_prob[k] for k, v1, v2 in grid.T}

    # Generate many samples via MCMC.
    total_count = 2000
    counts = defaultdict(lambda: 0)
    for seed in range(total_count):
        edges = sample_tree(grid, edge_prob, edges, seed)
        counts[tuple(edges)] += 1
    assert len(counts) == NUM_SPANNING_TREES[V]

    # Check accuracy using Pearson's chi-squared test.
    keys = counts.keys()
    counts = np.array([counts[key] for key in keys])
    probs = np.array(
        [np.prod([edge_prob_dict[edge] for edge in key]) for key in keys])
    probs /= probs.sum()
    assert 1e-2 < multinomial_goodness_of_fit(probs, counts, total_count)
