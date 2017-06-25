from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pytest

from treecat.serving import serve_model
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import train_model


@pytest.fixture(scope='module')
def model():
    return train_model(TINY_DATA, TINY_MASK, TINY_CONFIG)


@pytest.mark.parametrize('engine', [
    pytest.mark.xfail('numpy'),
    'tensorflow',
    pytest.mark.xfail('cython'),
])
def test_server_init(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)
    server._get_session(7)


@pytest.mark.parametrize('engine', [
    pytest.mark.xfail('numpy'),
    'tensorflow',
    pytest.mark.xfail('cython'),
])
def test_server_sample_shape(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)

    # Sample all possible mask patterns.
    N, V = TINY_DATA.shape
    factors = [[True, False]] * V
    for mask in itertools.product(*factors):
        mask = np.array(mask, dtype=np.bool_)
        samples = server.sample(TINY_DATA, mask)
        assert samples.shape == TINY_DATA.shape
        assert samples.dtype == TINY_DATA.dtype
        assert np.allclose(samples[:, mask], TINY_DATA[:, mask])


@pytest.mark.parametrize('engine', [
    pytest.mark.xfail('numpy'),
    pytest.mark.xfail('tensorflow'),
    pytest.mark.xfail('cython'),
])
def test_server_logprob_shape(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)

    # Sample all possible mask patterns.
    N, V = TINY_DATA.shape
    factors = [[True, False]] * V
    for mask in itertools.product(*factors):
        mask = np.array(mask, dtype=np.bool_)
        logprob = server.logprob(TINY_DATA, mask)
        assert logprob.shape == (N, )
        assert np.isfinite(logprob).all()
        assert (logprob < 0.0).all()  # Assuming features are discrete.


@pytest.mark.parametrize('engine', [
    pytest.mark.xfail('numpy'),
    pytest.mark.xfail('tensorflow'),
    pytest.mark.xfail('cython'),
])
def test_server_logprob_is_normalized(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)

    # The total probability of all possible rows should be 1.
    C = config['num_categories']
    N, V = TINY_DATA.shape
    factors = [range(C)] * V
    data = np.array(list(itertools.product(*factors)), dtype=np.int32)
    mask = np.array([True] * V, dtype=np.bool_)
    logprob = server.logprob(data, mask)
    logtotal = np.logaddexp.reduce(logprob)
    assert abs(logtotal) < 1e-6, logtotal
