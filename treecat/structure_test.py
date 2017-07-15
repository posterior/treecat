from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from collections import Counter
from collections import defaultdict

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.structure import OP_DOWN
from treecat.structure import OP_IN
from treecat.structure import OP_OUT
from treecat.structure import OP_ROOT
from treecat.structure import OP_UP
from treecat.structure import estimate_tree
from treecat.structure import find_center_of_tree
from treecat.structure import find_complete_edge
from treecat.structure import make_complete_graph
from treecat.structure import make_propagation_program
from treecat.structure import make_tree
from treecat.structure import print_tree
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
def test_make_propagation_program(edges, root):
    E = len(edges)
    V = E + 1
    grid = make_tree(edges)
    neighbors = {v: set() for v in range(V)}
    for e, v1, v2 in grid.T:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)

    # Generate a program.
    program = make_propagation_program(grid, root)
    assert program.shape == (V + E + 1 + E + V, 4)
    assert program.dtype == np.int16
    print(program)

    # Check that program edges are consistent with topology.
    if root is not None:
        assert program[V + E][0] == OP_ROOT
        assert program[V + E][1] == root
    else:
        root = program[V + E][1]
    assert set(row[1] for row in program) == set(range(V)), 'bad vertex set'
    for op, v, v2, e in program:
        if op == OP_IN or op == OP_OUT:
            assert v != v2
            assert v2 in neighbors[v]
            assert grid[1, e] == min(v, v2)
            assert grid[2, e] == max(v, v2)

    # Check that each instruction appears exactly once.
    op_counts = defaultdict(Counter)
    for op, v, v2, e in program:
        op_counts[op][v] += 1
    assert len(op_counts[OP_UP]) == V
    assert len(op_counts[OP_ROOT]) == 1
    assert len(op_counts[OP_OUT]) == E
    assert len(op_counts[OP_DOWN]) == V
    assert sum(op_counts[OP_UP].values()) == V
    assert sum(op_counts[OP_IN].values()) == E
    assert sum(op_counts[OP_ROOT].values()) == 1
    assert sum(op_counts[OP_OUT].values()) == E
    assert sum(op_counts[OP_DOWN].values()) == V

    # Check inward ordering.
    state = np.zeros(V, np.int8)
    for op, v, v2, e in program:
        if op == OP_UP:
            assert state[v] == 0
            state[v] = 1
        elif op == OP_IN:
            assert state[v] in (1, 2)
            assert state[v2] in (1, 2)
            state[v] = 2
        elif op == OP_ROOT:
            assert state[v] in (1, 2)
            state[v] = 3
        elif op == OP_OUT:
            assert state[v] in (1, 2, 3)
            assert state[v2] == 4
            state[v] = 3
        elif op == OP_DOWN:
            assert state[v] == 3
            state[v] = 4
    assert np.all(state == 4)


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
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
    num_samples = 30 * NUM_SPANNING_TREES[V]
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

    # Possibly truncate.
    T = 100
    truncated = False
    if len(counts) > T:
        counts = counts[:T]
        probs = probs[:T]
        truncated = True

    gof = multinomial_goodness_of_fit(
        probs, counts, num_samples, plot=True, truncated=truncated)
    assert 1e-2 < gof


def permute_tree(perm, tree):
    return tuple(sorted(tuple(sorted([perm[u], perm[v]])) for (u, v) in tree))


def iter_permuted_trees(vertices, tree):
    return set(
        permute_tree(perm, tree) for perm in itertools.permutations(vertices))


def close_under_permutations(V, tree_generators):
    vertices = list(range(V))
    return set.union(
        * [iter_permuted_trees(vertices, tree) for tree in tree_generators])


# These topologically distinct sets of trees generate sets of all trees
# under permutation of vertices.
TREE_GENERATORS = [
    [[]],
    [[]],
    [[(0, 1)]],
    [[(0, 1), (0, 2)]],
    [
        [(0, 1), (0, 2), (0, 3)],
        [(0, 1), (1, 2), (2, 3)],
    ],
    [
        [(0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 1), (0, 2), (0, 3), (1, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    ],
]


def get_spanning_trees(V):
    """Compute set of spanning trees on V vertices."""
    all_trees = close_under_permutations(V, TREE_GENERATORS[V])
    assert len(all_trees) == NUM_SPANNING_TREES[V]
    return all_trees


@pytest.mark.parametrize('num_edges', range(1, 10))
def test_estimate_tree(num_edges):
    set_random_seed(0)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = np.random.random([K]) - 0.5
    edges = estimate_tree(grid, edge_logits)

    # Check size.
    assert len(edges) == E
    for v in range(V):
        assert any(v in edge for edge in edges)

    # Check optimality.
    edges = tuple(edges)
    if V < len(TREE_GENERATORS):
        all_trees = get_spanning_trees(V)
        assert edges in all_trees
        all_trees = list(all_trees)
        logits = []
        for tree in all_trees:
            logits.append(
                sum(edge_logits[find_complete_edge(u, v)] for (u, v) in tree))
        expected = all_trees[np.argmax(logits)]
        assert edges == expected


@pytest.mark.parametrize('edges', sum(TREE_GENERATORS, []))
def test_print_tree(edges):
    V = 1 + len(edges)
    feature_names = ['feature-{}'.format(v) for v in range(V)]
    for root in feature_names:
        print('-' * 32)
        text = print_tree(edges, feature_names, root)
        print(text)
        assert len(text.split('\n')) == V
