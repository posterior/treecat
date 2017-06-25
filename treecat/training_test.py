from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from treecat.structure import TreeStructure
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import get_annealing_schedule
from treecat.training import train_model


def test_get_annealing_schedule():
    np.random.seed(0)
    num_rows = 10
    schedule = get_annealing_schedule(num_rows, TINY_CONFIG)
    for step, (action, row_id) in enumerate(schedule):
        assert step < 1000
        assert action in ['add_row', 'remove_row', 'batch']
        if action == 'batch':
            assert row_id is None
        else:
            assert 0 <= row_id and row_id < num_rows


@pytest.mark.parametrize('engine', [
    'numpy',
    'tensorflow',
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
    C = config['num_categories']
    M = config['num_clusters']
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
