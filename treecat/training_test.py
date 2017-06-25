from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import get_annealing_schedule
from treecat.training import train_model


def test_get_annealing_schedule():
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
    pytest.mark.xfail('cython'),
])
def test_train_model(engine):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    N, V = TINY_DATA.shape
    E = V - 1
    model = train_model(TINY_DATA, TINY_MASK, config)
    assert model['config'] == config
    assert model['suffstats']['feat_ss'].sum() == TINY_MASK.sum()
    assert model['suffstats']['vert_ss'].sum() == N * V
    assert model['suffstats']['edge_ss'].sum() == N * E
    # TODO Check mores suffstats invariants.
