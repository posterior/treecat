from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import pytest

from treecat.config import DEFAULT_CONFIG
from treecat.structure import TreeStructure
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
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


@pytest.mark.parametrize('engine', [
    'numpy',
])
def test_train_model(engine):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    model = train_model(TINY_DATA, TINY_MASK, config)

    assert model['config'] == config
    assert isinstance(model['tree'], TreeStructure)
    grid = model['tree'].tree_grid
    feat_ss = model['suffstats']['feat_ss']
    vert_ss = model['suffstats']['vert_ss']
    edge_ss = model['suffstats']['edge_ss']
    assignments = model['assignments']

    # Check shape.
    N, V = TINY_DATA.shape
    E = V - 1
    C = config['model_num_categories']
    M = config['model_num_clusters']
    assert grid.shape == (3, E)
    assert feat_ss.shape == (V, C, M)
    assert vert_ss.shape == (V, M)
    assert edge_ss.shape == (E, M, M)
    assert assignments.shape == (N, V)

    # Check bounds.
    assert np.all(0 <= feat_ss)
    assert np.all(0 <= vert_ss)
    assert np.all(0 <= edge_ss)
    assert np.all(0 <= assignments)
    assert np.all(assignments < M)

    # Check totals.
    assert feat_ss.sum() == TINY_MASK.sum()
    assert np.all(feat_ss.sum((1, 2)) == TINY_MASK.sum(0))
    assert vert_ss.sum() == N * V
    assert np.all(vert_ss.sum(1) == N)
    assert edge_ss.sum() == N * E
    assert np.all(edge_ss.sum((1, 2)) == N)

    # Check marginals.
    assert np.all(feat_ss.sum(1) <= vert_ss)
    assert np.all(edge_ss.sum(2) == vert_ss[grid[1, :]])
    assert np.all(edge_ss.sum(1) == vert_ss[grid[2, :]])


def generate_tiny_dataset(num_rows, num_cols, num_cats):
    np.random.seed(0)
    shape = (num_rows, num_cols)
    data = np.random.randint(num_cats, size=shape, dtype=np.int32)
    mask = np.ones(shape, dtype=np.bool_)
    return data, mask


def hash_assignments(assignments):
    assert isinstance(assignments, np.ndarray)
    return tuple(tuple(row) for row in assignments)


@pytest.mark.parametrize('N,V,C,M', [
    (2, 2, 2, 2),
    (2, 2, 2, 3),
    (2, 3, 2, 2),
    (3, 2, 2, 2),
])
def test_category_sampler(N, V, C, M):
    config = DEFAULT_CONFIG.copy()
    config['learning_sample_tree_steps'] = 0  # Disable tree kernel.
    config['model_num_categories'] = C
    config['model_num_clusters'] = M
    data, mask = generate_tiny_dataset(num_rows=N, num_cols=V, num_cats=C)
    trainer = TreeCatTrainer(data, mask, config)
    print('Data:')
    print(data)

    # Add all rows.
    for row_id in range(N):
        trainer.add_row(row_id)

    # Collect samples.
    counts = defaultdict(lambda: 0)
    for _ in range(2000):
        for row_id in range(N):
            # This is a single-site Gibbs sampler.
            trainer.remove_row(row_id)
            trainer.add_row(row_id)
        counts[hash_assignments(trainer.assignments)] += 1
    print('Count\tAssignments')
    for key, count in sorted(counts.items()):
        print('{}\t{}'.format(count, key))
    assert len(counts) == M**(N * V)
