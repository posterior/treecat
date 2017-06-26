from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from collections import defaultdict

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.serving import make_posterior
from treecat.serving import serve_model
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import train_model


@pytest.fixture(scope='module')
def model():
    return train_model(TINY_DATA, TINY_MASK, TINY_CONFIG)


def test_make_posterior(model):
    grid = model['tree'].tree_grid
    suffstats = model['suffstats']
    factors = make_posterior(grid, suffstats)
    observed = factors['observed']
    observed_latent = factors['observed_latent']
    latent = factors['latent']
    latent_latent = factors['latent_latent']

    # Check shape.
    N, V = TINY_DATA.shape
    E = V - 1
    C = TINY_CONFIG['num_categories']
    M = TINY_CONFIG['num_clusters']
    assert observed.shape == (V, C)
    assert observed_latent.shape == (V, C, M)
    assert latent.shape == (V, M)
    assert latent_latent.shape == (E, M, M)

    # Check normalization.
    atol = 1e-5
    assert np.allclose(observed.sum(1), 1.0, atol=atol)
    assert np.allclose(observed_latent.sum((1, 2)), 1.0, atol=atol)
    assert np.allclose(latent.sum(1), 1.0, atol=atol)
    assert np.allclose(latent_latent.sum((1, 2)), 1.0, atol=atol)

    # Check marginals.
    assert np.allclose(observed_latent.sum(2), observed, atol=atol)
    assert np.allclose(observed_latent.sum(1), latent, atol=atol)
    assert np.allclose(latent_latent.sum(2), latent[grid[1, :], :], atol=atol)
    assert np.allclose(latent_latent.sum(1), latent[grid[2, :], :], atol=atol)


@pytest.mark.parametrize('engine', [
    'numpy',
    'tensorflow',
])
def test_server_init(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    serve_model(model['tree'], model['suffstats'], config)


@pytest.mark.parametrize('engine', [
    'numpy',
    'tensorflow',
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
    'numpy',
    'tensorflow',
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


@pytest.mark.parametrize('engine', [
    'numpy',
    pytest.mark.xfail('tensorflow'),
])
def test_server_logprob_negative(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)

    # Sample all possible mask patterns.
    N, V = TINY_DATA.shape
    factors = [[True, False]] * V
    for mask in itertools.product(*factors):
        mask = np.array(mask, dtype=np.bool_)
        logprob = server.logprob(TINY_DATA, mask)
        abstol = 1e-5
        assert (logprob <= abstol).all()  # Assuming features are discrete.


@pytest.mark.parametrize('engine', [
    'numpy',
    pytest.mark.xfail('tensorflow'),
])
def test_server_logprob_normalized(engine, model):
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
    assert logtotal == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize('engine', [
    pytest.mark.xfail('numpy'),
    pytest.mark.xfail('tensorflow'),
])
def test_server_gof(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)

    # Generate samples.
    N = 20000  # Number of samples.
    C = config['num_categories']
    V = TINY_DATA.shape[1]
    empty_data = np.zeros([N, V], dtype=np.int32)
    empty_mask = np.array([False] * V, dtype=np.bool_)
    full_mask = np.array([True] * V, dtype=np.bool_)
    samples = server.sample(empty_data, empty_mask)
    logprob = server.logprob(samples, full_mask)

    # Check that each row was sampled at least once.
    counts = defaultdict(lambda: 0)
    probs = {}
    for row_data, row_prob in zip(samples, np.exp(logprob)):
        row_data = tuple(row_data)
        counts[row_data] += 1
        probs[row_data] = row_prob
    keys = sorted(counts.keys())
    assert len(keys) == C**V

    # Check accuracy using Pearson's chi-squared test.
    counts = np.array([counts[key] for key in keys])
    probs = np.array([probs[key] for key in keys])
    probs /= probs.sum()  # Test normalization elsewhere.
    assert 1e-2 < multinomial_goodness_of_fit(probs, counts, total_count=N)


@pytest.mark.parametrize('engine', [
    'numpy',
    pytest.mark.xfail('tensorflow'),
])
def test_server_entropy(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)
    V = TINY_DATA.shape[1]
    feature_sets = [(v1, v2) for v2 in range(V) for v1 in range(v2)]
    entropies = server.entropy(feature_sets)
    assert entropies.shape == (len(feature_sets), )
    assert np.all(np.isfinite(entropies))
    assert np.all(entropies >= 0)


@pytest.mark.parametrize('engine', [
    'numpy',
    'tensorflow',
])
def test_server_correlation(engine, model):
    config = TINY_CONFIG.copy()
    config['engine'] = engine
    server = serve_model(model['tree'], model['suffstats'], config)
    V = TINY_DATA.shape[1]
    correlation = server.correlation()
    assert correlation.shape == (V, V)
    assert np.all(np.isfinite(correlation))
    assert np.allclose(correlation, correlation.T)
    assert np.all(0 <= correlation)
    assert np.all(correlation <= 1.0)
