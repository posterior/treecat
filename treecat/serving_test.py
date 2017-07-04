from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.generate import generate_fake_model
from treecat.serving import serve_model
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_RAGGED_INDEX
from treecat.testutil import numpy_seterr
from treecat.training import train_model
from treecat.util import set_random_seed

numpy_seterr()


@pytest.fixture(scope='module')
def model():
    return train_model(TINY_RAGGED_INDEX, TINY_DATA, TINY_CONFIG)


def test_server_init(model):
    serve_model(model['tree'], model['suffstats'], TINY_CONFIG)


def test_server_sample_shape(model):
    ragged_index = TINY_RAGGED_INDEX
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample many different counts patterns.
    V = len(ragged_index) - 1
    N = data.shape[0]
    factors = [[0, 1, 2]] * V
    for counts in itertools.product(*factors):
        counts = np.array(counts, dtype=np.int8)
        for n in range(N):
            row = data[n, :]
            samples = server.sample(N, counts, row)
            assert samples.shape == (N, row.shape[0])
            assert samples.dtype == row.dtype
            for v in range(V):
                beg, end = ragged_index[v:v + 2]
                assert np.all(samples[:, beg:end].sum(axis=1) == counts[v])


def test_server_logprob_shape(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)
    logprobs = server.logprob(data)
    N = data.shape[0]
    assert logprobs.dtype == np.float32
    assert logprobs.shape == (N, )
    assert np.isfinite(logprobs).all()


def one_hot(c, C):
    value = np.zeros(C, dtype=np.int8)
    value[c] = 1
    return value


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 1, 2, 3),
    (10, 2, 2, 2),
    pytest.mark.xfail((10, 2, 2, 3)),
    pytest.mark.xfail((10, 3, 2, 2)),
    pytest.mark.xfail((10, 4, 2, 2)),
])
def test_server_logprob_normalized(N, V, C, M):
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    server = serve_model(model['tree'], model['suffstats'], config)

    # The total probability of all categorical rows should be 1.
    ragged_index = model['suffstats']['ragged_index']
    factors = []
    for v in range(V):
        C = ragged_index[v + 1] - ragged_index[v]
        factors.append([one_hot(c, C) for c in range(C)])
    data = np.array(
        [np.concatenate(columns) for columns in itertools.product(*factors)],
        dtype=np.int8)
    logprobs = server.logprob(data)
    logtotal = np.logaddexp.reduce(logprobs)
    assert logtotal == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 1, 2, 3),
    (10, 2, 2, 2),
    (10, 3, 2, 2),
    (10, 4, 2, 2),
    (20, 1, 2, 2),
    (20, 1, 2, 3),
    (20, 2, 2, 2),
    (20, 3, 2, 2),
    (20, 4, 2, 2),
    (40, 1, 2, 2),
    (40, 1, 2, 3),
    (40, 2, 2, 2),
    (40, 3, 2, 2),
    (40, 4, 2, 2),
])
def test_server_unconditional_gof(N, V, C, M):
    set_random_seed(0)
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    server = serve_model(model['tree'], model['suffstats'], config)

    # Generate samples.
    expected = C**V
    num_samples = 1000 * expected
    ones = np.ones(V, dtype=np.int8)
    samples = server.sample(num_samples, ones)
    logprobs = server.logprob(samples)
    counts = {}
    probs = {}
    for sample, logprob in zip(samples, logprobs):
        key = tuple(sample)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
            probs[key] = np.exp(logprob)
    assert len(counts) == expected

    # Check accuracy using Pearson's chi-squared test.
    keys = sorted(counts.keys(), key=lambda key: -probs[key])
    counts = np.array([counts[k] for k in keys], dtype=np.int32)
    probs = np.array([probs[k] for k in keys])
    probs /= probs.sum()
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 1, 2, 3),
    (10, 2, 2, 2),
    (10, 3, 2, 2),
    (10, 4, 2, 2),
    (20, 1, 2, 2),
    (20, 1, 2, 3),
    (20, 2, 2, 2),
    (20, 3, 2, 2),
    (20, 4, 2, 2),
    (40, 1, 2, 2),
    (40, 1, 2, 3),
    (40, 2, 2, 2),
    (40, 3, 2, 2),
    (40, 4, 2, 2),
])
def test_server_conditional_gof(N, V, C, M):
    set_random_seed(0)
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    server = serve_model(model['tree'], model['suffstats'], config)

    # Generate samples.
    expected = C**V
    num_samples = 1000 * expected
    ones = np.ones(V, dtype=np.int8)
    cond_data = server.sample(1, ones).reshape(server.zero_row().shape)
    samples = server.sample(num_samples, ones, cond_data)
    logprobs = server.logprob(samples + cond_data[np.newaxis, :])
    counts = {}
    probs = {}
    for sample, logprob in zip(samples, logprobs):
        key = tuple(sample)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
            probs[key] = np.exp(logprob)
    assert len(counts) == expected

    # Check accuracy using Pearson's chi-squared test.
    keys = sorted(counts.keys(), key=lambda key: -probs[key])
    counts = np.array([counts[k] for k in keys], dtype=np.int32)
    probs = np.array([probs[k] for k in keys])
    probs /= probs.sum()

    truncated = False
    valid = (probs * num_samples > 20)
    if not valid.all():
        T = valid.argmin()
        probs = probs[:T]
        counts = counts[:T]
        truncated = True

    gof = multinomial_goodness_of_fit(
        probs, counts, num_samples, plot=True, truncated=truncated)
    assert 1e-2 < gof


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 2, 2, 3),
    (10, 3, 2, 4),
    (10, 4, 2, 5),
    (10, 5, 2, 6),
    (10, 6, 2, 7),
    (10, 7, 2, 8),
])
def test_correlation(N, V, C, M):
    set_random_seed(0)
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    server = serve_model(model['tree'], model['suffstats'], config)

    correlation = server.correlation()
    print(correlation)
    assert np.all(0 <= correlation)
    assert np.all(correlation <= 1)
    assert np.allclose(correlation, correlation.T)
    for v in range(V):
        assert correlation[v, :].argmax() == v
        assert correlation[:, v].argmax() == v
