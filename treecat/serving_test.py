from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.generate import generate_fake_model
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


def test_server_sample_shape(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample many different counts patterns.
    V = len(data)
    N = data[0].shape[0]
    factors = [[0, 1, 2]] * V
    for counts in itertools.product(*factors):
        counts = np.array(counts, dtype=np.int8)
        for n in range(N):
            row = [col[n, :] for col in data]
            sample = server.sample(row, counts)
            assert len(sample) == len(row)
            for v in range(V):
                assert sample[v].shape == row[v].shape
                assert sample[v].dtype == row[v].dtype
                assert np.all(sample[v].sum() == counts[v])


def test_server_logprob_runs(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample all possible mask patterns.
    N = data[0].shape[0]
    abstol = 1e-5
    for n in range(N):
        row = [col[n, :] for col in data]
        logprob = server.logprob(row)
        assert isinstance(logprob, float)
        assert np.isfinite(logprob)
        assert logprob < abstol


def one_hot(c, C):
    value = np.zeros(C, dtype=np.int8)
    value[c] = 1
    return value


def test_server_logprob_normalized(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # The total probability of all categorical rows should be 1.
    V = len(data)
    factors = []
    for v in range(V):
        C = data[v].shape[1]
        factors.append([one_hot(c, C) for c in range(C)])
    logprobs = []
    for row in itertools.product(*factors):
        logprobs.append(server.logprob(row))
    logtotal = np.logaddexp.reduce(logprobs)
    assert logtotal == pytest.approx(0.0, abs=1e-5)


def hash_row(row):
    return tuple(tuple(col) for col in row)


@pytest.mark.parametrize('N,V,C,M', [
    (1, 1, 2, 2),
    (1, 1, 2, 3),
    (1, 2, 2, 2),
    (1, 3, 2, 2),
    pytest.mark.xfail((1, 4, 2, 2)),
    (2, 1, 2, 2),
    pytest.mark.xfail((2, 1, 2, 3)),
    (2, 2, 2, 2),
    (2, 3, 2, 2),
    pytest.mark.xfail((2, 4, 2, 2)),
    (4, 1, 2, 2),
    pytest.mark.xfail((4, 1, 2, 3)),
    (4, 2, 2, 2),
    (4, 3, 2, 2),
    pytest.mark.xfail((4, 4, 2, 2)),
])
def test_server_gof(N, V, C, M):
    np.random.seed(0)
    data, model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    server = serve_model(model['tree'], model['suffstats'], config)

    # Generate samples.
    cond_data = [np.zeros(C, np.int8) for col in data]
    expected = C**V
    num_samples = 100 * expected
    ones = np.ones(len(data), dtype=np.int8)
    counts = {}
    logprobs = {}
    for _ in range(num_samples):
        sample = server.sample(cond_data, ones)
        key = hash_row(sample)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
            logprobs[key] = server.logprob(sample)
    assert len(counts) == expected

    # Check accuracy using Pearson's chi-squared test.
    keys = sorted(counts.keys(), key=lambda key: -logprobs[key])
    counts = np.array([counts[k] for k in keys], dtype=np.int32)
    probs = np.exp(np.array([logprobs[k] for k in keys]))
    probs /= probs.sum()
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof
