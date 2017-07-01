from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.config import DEFAULT_CONFIG
from treecat.generate import generate_dataset
from treecat.structure import TreeStructure
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.training import TreeCatTrainer
from treecat.training import get_annealing_schedule
from treecat.training import train_model


def test_get_annealing_schedule():
    np.random.seed(0)
    num_rows = 10
    schedule = get_annealing_schedule(num_rows, TINY_CONFIG)
    for step, (action, row_id) in enumerate(schedule):
        assert step < 1000
        assert action in ['add_row', 'remove_row', 'sample_tree']
        if action == 'sample_tree':
            assert row_id is None
        else:
            assert 0 <= row_id and row_id < num_rows


def test_train_model():
    data = TINY_DATA
    config = TINY_CONFIG
    model = train_model(data, config)

    assert model['config'] == config
    assert isinstance(model['tree'], TreeStructure)
    grid = model['tree'].tree_grid
    assignments = model['assignments']
    vert_ss = model['suffstats']['vert_ss']
    edge_ss = model['suffstats']['edge_ss']
    feat_ss = model['suffstats']['feat_ss']

    # Check shape.
    V = len(data)
    N = data[0].shape[0]
    E = V - 1
    M = config['model_num_clusters']
    assert grid.shape == (3, E)
    assert assignments.shape == (N, V)
    assert vert_ss.shape == (V, M)
    assert edge_ss.shape == (E, M, M)
    assert len(feat_ss) == V
    for v in range(V):
        assert feat_ss[v].shape == (data[v].shape[1], M)

    # Check bounds.
    assert np.all(assignments < M)
    assert np.all(0 <= vert_ss)
    assert np.all(0 <= edge_ss)
    assert np.all(0 <= assignments)
    for v in range(V):
        assert np.all(0 <= feat_ss[v])

    # Check totals.
    assert vert_ss.sum() == N * V
    assert np.all(vert_ss.sum(1) == N)
    assert edge_ss.sum() == N * E
    assert np.all(edge_ss.sum((1, 2)) == N)
    for v in range(V):
        assert feat_ss[v].sum() == data[v].sum()

    # Check marginals.
    assert np.all(edge_ss.sum(2) == vert_ss[grid[1, :]])
    assert np.all(edge_ss.sum(1) == vert_ss[grid[2, :]])


def hash_assignments(assignments):
    assert isinstance(assignments, np.ndarray)
    return tuple(tuple(row) for row in assignments)


@pytest.mark.parametrize('N,V,C,M', [
    (1, 1, 1, 1),
    (1, 2, 2, 2),
    (1, 2, 2, 3),
    (1, 3, 2, 2),
    (1, 4, 2, 2),
    (2, 1, 2, 2),
    pytest.mark.xfail((2, 1, 2, 3)),
    pytest.mark.xfail((2, 2, 1, 2)),
    pytest.mark.xfail((2, 2, 2, 2)),
    pytest.mark.xfail((2, 3, 2, 2)),
    (3, 1, 2, 2),
    (4, 1, 1, 2),
])
def test_assignment_sampler_gof(N, V, C, M):
    config = DEFAULT_CONFIG.copy()
    config['learning_sample_tree_steps'] = 0  # Disable tree kernel.
    config['model_num_clusters'] = M
    data = generate_dataset(num_rows=N, num_cols=V, num_cats=C)
    trainer = TreeCatTrainer(data, config)
    print('Data:')
    for col in data:
        print(col)

    # Add all rows.
    for row_id in range(N):
        trainer.add_row(row_id)

    # Collect samples.
    num_samples = 100 * M**(N * V)
    counts = {}
    logprobs = {}
    for _ in range(num_samples):
        for row_id in range(N):
            # This is a single-site Gibbs sampler.
            trainer.remove_row(row_id)
            trainer.add_row(row_id)
        key = hash_assignments(trainer.assignments)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
            logprobs[key] = trainer.logprob()
    print('logprobs:')
    for key, val in sorted(logprobs.items()):
        print('{:0.3g}\t{}'.format(val, key))
    assert len(counts) == M**(N * V)

    # Check accuracy using Pearson's chi-squared test.
    keys = counts.keys()
    counts = np.array([counts[k] for k in keys], dtype=np.int32)
    probs = np.exp(np.array([logprobs[k] for k in keys]))
    probs /= probs.sum()
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof
