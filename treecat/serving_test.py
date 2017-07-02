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
from treecat.training import train_model


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
            sample = server.sample(counts, row)
            assert sample.shape == row.shape
            assert sample.dtype == row.dtype
            for v in range(V):
                beg, end = ragged_index[v:v + 2]
                assert np.all(sample[beg:end].sum() == counts[v])


def test_server_logprob_runs(model):
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # Sample all possible mask patterns.
    N = data.shape[0]
    for n in range(N):
        logprob = server.logprob(data[n, :])
        assert isinstance(logprob, float)
        assert np.isfinite(logprob)


def one_hot(c, C):
    value = np.zeros(C, dtype=np.int8)
    value[c] = 1
    return value


@pytest.mark.xfail
def test_server_logprob_normalized(model):
    ragged_index = TINY_RAGGED_INDEX
    data = TINY_DATA
    server = serve_model(model['tree'], model['suffstats'], TINY_CONFIG)

    # The total probability of all categorical rows should be 1.
    V = len(data)
    factors = []
    for v in range(V):
        C = ragged_index[v + 1] - ragged_index[v]
        factors.append([one_hot(c, C) for c in range(C)])
    logprobs = []
    row = server.zero_row
    for columns in itertools.product(*factors):
        for v, column in enumerate(columns):
            beg, end = ragged_index[v:v + 2]
            row[beg:end] = column
        logprobs.append(server.logprob(row))
    logtotal = np.logaddexp.reduce(logprobs)
    assert logtotal == pytest.approx(0.0, abs=1e-5)


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
    expected = C**V
    num_samples = 100 * expected
    ones = np.ones(V, dtype=np.int8)
    counts = {}
    logprobs = {}
    for _ in range(num_samples):
        sample = server.sample(ones)
        key = tuple(sample)
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
