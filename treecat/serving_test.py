from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pytest

from treecat.serving import TreeCatServer
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import train_model


@pytest.fixture(scope='module')
def model():
    return train_model(TINY_DATA, TINY_MASK, TINY_CONFIG)


def test_server_init(model):
    server = TreeCatServer(model['tree'], model['suffstats'], TINY_CONFIG)
    server._get_session(7)


def test_server_sample_shape(model):
    N, V = TINY_DATA.shape
    server = TreeCatServer(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample all possible mask patterns.
    factors = [[True, False]] * V
    for mask in itertools.product(*factors):
        mask = np.array(mask, dtype=np.bool_)
        samples = server.sample(TINY_DATA, mask)
        assert samples.shape == TINY_DATA.shape
        assert samples.dtype == TINY_DATA.dtype
        assert np.allclose(samples[:, mask], TINY_DATA[:, mask])


def test_server_logprob_shape(model):
    N, V = TINY_DATA.shape
    server = TreeCatServer(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample all possible mask patterns.
    factors = [[True, False]] * V
    for mask in itertools.product(*factors):
        mask = np.array(mask, dtype=np.bool_)
        logprob = server.logprob(TINY_DATA, mask)
        assert logprob.shape == (N, )
        assert np.isfinite(logprob).all()
        assert (logprob < 0.0).all()  # Assuming features are discrete.


@pytest.mark.xfail
def test_server_logprob_is_normalized(model):
    N, V = TINY_DATA.shape
    C = TINY_CONFIG['num_categories']
    server = TreeCatServer(model['tree'], model['suffstats'], TINY_CONFIG)

    # The total probability of all possible rows should be 1.
    factors = [range(C)] * V
    data = np.array(list(itertools.product(*factors)), dtype=np.int32)
    mask = np.array([True] * V, dtype=np.bool_)
    logprob = server.logprob(data, mask)
    total = np.exp(np.logaddexp.reduce(logprob))
    assert abs(total - 1.0) < 1e-6, total
