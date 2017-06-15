from __future__ import absolute_import, division, print_function

from collections import defaultdict

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.rst import sample_tree_exact_naive
from treecat.structure import make_complete_graph
from treecat.util import NUM_SPANNING_TREES


@pytest.mark.parametrize('num_vertices', [
    1,
    2,
    pytest.mark.xfail(3),
    pytest.mark.xfail(4),
])
def test_sample_tree_exact_naive(num_vertices):
    np.random.seed(num_vertices)
    V, K, grid = make_complete_graph(num_vertices)
    edge_prob = np.exp(-np.random.uniform(size=[K]))
    edge_prob_dict = {(v1, v2): edge_prob[k] for k, v1, v2 in grid.T}

    # Generate many samples.
    total_count = 2000
    counts = defaultdict(lambda: 0)
    for seed in range(total_count):
        edges = sample_tree_exact_naive(V, grid, edge_prob, seed)
        counts[tuple(edges)] += 1
    assert len(counts) == NUM_SPANNING_TREES[V]

    # Check accuracy using Pearson's chi-squared test.
    keys = counts.keys()
    counts = np.array([counts[key] for key in keys])
    probs = np.array(
        [np.prod([edge_prob_dict[edge] for edge in key]) for key in keys])
    probs /= probs.sum()
    assert 1e-2 < multinomial_goodness_of_fit(probs, counts, total_count)
