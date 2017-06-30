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
from treecat.training import train_model


@pytest.fixture(scope='module')
def model():
    return train_model(TINY_DATA, TINY_CONFIG)


def test_make_posterior(model):
    data = TINY_DATA
    grid = model['tree'].tree_grid
    suffstats = model['suffstats']
    factors = make_posterior(grid, suffstats)
    observed = factors['observed']
    observed_latent = factors['observed_latent']
    latent = factors['latent']
    latent_latent = factors['latent_latent']

    # Check shape.
    V = len(data)
    E = V - 1
    M = TINY_CONFIG['model_num_clusters']
    assert latent.shape == (V, M)
    assert latent_latent.shape == (E, M, M)
    assert len(observed) == V
    assert len(observed_latent) == V
    for v in range(V):
        assert len(observed[v].shape) == 1
        C = observed[v].shape[0]
        assert observed_latent[v].shape == (C, M)

    # Check normalization.
    atol = 1e-5
    assert np.allclose(latent.sum(1), 1.0, atol=atol)
    assert np.allclose(latent_latent.sum((1, 2)), 1.0, atol=atol)
    for v in range(V):
        assert np.allclose(observed[v].sum(), 1.0, atol=atol)
        assert np.allclose(observed_latent[v].sum(), 1.0, atol=atol)

    # Check marginals.
    assert np.allclose(latent_latent.sum(2), latent[grid[1, :], :], atol=atol)
    assert np.allclose(latent_latent.sum(1), latent[grid[2, :], :], atol=atol)
    for v in range(V):
        assert np.allclose(observed_latent[v].sum(1), observed[v], atol=atol)
        assert np.allclose(observed_latent[v].sum(0), latent[v], atol=atol)


def test_server_init(model):
    serve_model(model['tree'], model['suffstats'], TINY_CONFIG)


@pytest.mark.xfail
def test_server_sample_shape(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample many different counts patterns.
    V = len(data)
    factors = [[0, 1, 2]] * V
    for counts in itertools.product(*factors):
        counts = np.array(counts, dtype=np.int8)
        samples = server.sample(data, counts)
        assert len(samples) == len(data)
        for v in range(V):
            assert samples[v].shape == data[v].shape
            assert samples[v].dtype == data[v].dtype
            assert np.all(samples[v].sum(axis=1) == counts[v])


@pytest.mark.xfail
def test_server_logprob_runs(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample all possible mask patterns.
    V = len(data)
    N = data[0].shape[0]
    abstol = 1e-5
    factors = [[0, 1, 2]] * V
    for counts in itertools.product(*factors):
        counts = np.array(counts, dtype=np.int8)
        for n in range(N):
            row = [col[n, :] for col in data]
            logprob = server.logprob(row, counts)
            assert isinstance(logprob, float)
            assert np.isfinite(logprob)
            assert logprob < abstol


@pytest.mark.xfail
def test_server_logprob_normalized(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # The total probability of all rows should be 1.
    V = len(data)
    factors = [range(column.shape[1]) for column in data]
    pytest.mark.xfail(reason='TODO')
    data = np.array(list(itertools.product(*factors)), dtype=np.int8)
    mask = np.array([True] * V, dtype=np.bool_)
    logprob = server.logprob(data, mask)
    logtotal = np.logaddexp.reduce(logprob)
    assert abs(logtotal) < 1e-6, logtotal
    assert logtotal == pytest.approx(0.0, abs=1e-5)


@pytest.mark.xfail
def test_server_gof(model):
    np.random.seed(0)
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)
    num_samples = 50000

    # Generate samples.
    N = num_samples
    V = len(data)
    empty_data = np.zeros([N, V], dtype=np.int8)
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
    assert len(keys) == np.prod([col.shape[1] for col in data])

    # Check accuracy using Pearson's chi-squared test.
    counts = np.array([counts[key] for key in keys])
    probs = np.array([probs[key] for key in keys])
    probs /= probs.sum()  # Test normalization elsewhere.
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof
