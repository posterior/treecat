from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.generate import generate_dataset
from treecat.generate import generate_fake_ensemble
from treecat.generate import generate_fake_model
from treecat.serving import EnsembleServer
from treecat.serving import TreeCatServer
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_TABLE
from treecat.testutil import make_seed
from treecat.testutil import numpy_seterr
from treecat.testutil import xfail_param
from treecat.training import train_ensemble
from treecat.training import train_model
from treecat.util import set_random_seed

numpy_seterr()


@pytest.fixture(scope='module')
def model():
    V = TINY_TABLE.num_cols
    K = V * (V - 1) // 2
    tree_prior = np.zeros(K, np.float32)
    return train_model(TINY_TABLE, tree_prior, TINY_CONFIG)


@pytest.fixture(scope='module')
def ensemble():
    V = TINY_TABLE.num_cols
    K = V * (V - 1) // 2
    tree_prior = np.zeros(K, np.float32)
    return train_ensemble(TINY_TABLE, tree_prior, TINY_CONFIG)


def validate_sample_shape(table, server):
    # Sample many different counts patterns.
    V = table.num_cols
    N = table.num_rows
    factors = [[0, 1, 2]] * V
    for counts in itertools.product(*factors):
        counts = np.array(counts, dtype=np.int8)
        for n in range(N):
            row = table.data[n, :]
            samples = server.sample(N, counts, row)
            assert samples.shape == (N, row.shape[0])
            assert samples.dtype == row.dtype
            for v in range(V):
                beg, end = table.ragged_index[v:v + 2]
                assert np.all(samples[:, beg:end].sum(axis=1) == counts[v])


def test_server_sample_shape(model):
    server = TreeCatServer(model)
    validate_sample_shape(TINY_TABLE, server)


def test_ensemble_sample_shape(ensemble):
    server = EnsembleServer(ensemble)
    validate_sample_shape(TINY_TABLE, server)


def test_server_logprob_shape(model):
    table = TINY_TABLE
    server = TreeCatServer(model)
    logprobs = server.logprob(table.data)
    N = table.num_rows
    assert logprobs.dtype == np.float32
    assert logprobs.shape == (N, )
    assert np.isfinite(logprobs).all()


def test_ensemble_logprob_shape(ensemble):
    table = TINY_TABLE
    server = EnsembleServer(ensemble)
    logprobs = server.logprob(table.data)
    N = table.num_rows
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
    (10, 2, 2, 3),
    (10, 2, 3, 3),
    (10, 3, 2, 2),
    (10, 3, 3, 3),
    (10, 4, 2, 2),
    (10, 4, 3, 3),
    (10, 5, 2, 2),
    (10, 5, 3, 3),
    (10, 6, 2, 2),
    (10, 6, 3, 3),
])
def test_server_logprob_normalized(N, V, C, M):
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)

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
    (10, 2, 2, 2),
    (10, 2, 2, 3),
    (10, 3, 4, 5),
    (10, 4, 5, 6),
    (10, 5, 6, 7),
])
def test_server_marginals(N, V, C, M):
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)

    # Evaluate on random data.
    data = generate_dataset(N, V, C)['data']
    marginals = server.marginals(data)
    ragged_index = model['suffstats']['ragged_index']
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        totals = marginals[:, beg:end].sum(axis=1)
        assert np.allclose(totals, 1.0)


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 2, 2, 2),
    (10, 2, 2, 3),
    (10, 3, 4, 5),
    (10, 4, 5, 6),
    (10, 5, 6, 7),
])
def test_server_median(N, V, C, M):
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)

    # Evaluate on random data.
    counts = np.random.randint(10, size=[V], dtype=np.int8)
    data = generate_dataset(N, V, C)['data']
    median = server.median(counts, data)
    assert median.shape == data.shape
    assert median.dtype == np.int8
    ragged_index = model['suffstats']['ragged_index']
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        totals = median[:, beg:end].sum(axis=1)
        assert np.all(totals == counts[v])


NVCM_EXAMPLES_FOR_GOF = [
    # N, V, C, M.
    (10, 1, 2, 2),
    (10, 1, 2, 3),
    (10, 2, 2, 2),
    (10, 3, 2, 2),
    xfail_param(10, 4, 2, 2, reason='flaky'),
    (20, 1, 2, 2),
    (20, 1, 2, 3),
    (20, 2, 2, 2),
    (20, 3, 2, 2),
    (20, 4, 2, 2),
    (40, 1, 2, 2),
    (40, 1, 2, 3),
    (40, 2, 2, 2),
    (40, 3, 2, 2),
    xfail_param(40, 4, 2, 2, reason='flaky'),
]


@pytest.mark.parametrize('N,V,C,M', NVCM_EXAMPLES_FOR_GOF)
def test_server_unconditional_gof(N, V, C, M):
    set_random_seed(make_seed(N, V, C, M, 0))
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)
    validate_gof(N, V, C, M, server, conditional=False)


@pytest.mark.parametrize('N,V,C,M', NVCM_EXAMPLES_FOR_GOF)
def test_server_conditional_gof(N, V, C, M):
    set_random_seed(make_seed(N, V, C, M, 1))
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)
    validate_gof(N, V, C, M, server, conditional=True)


@pytest.mark.parametrize('N,V,C,M', NVCM_EXAMPLES_FOR_GOF)
def test_ensemble_unconditional_gof(N, V, C, M):
    ensemble = generate_fake_ensemble(N, V, C, M)
    server = EnsembleServer(ensemble)
    validate_gof(N, V, C, M, server, conditional=False)


@pytest.mark.parametrize('N,V,C,M', NVCM_EXAMPLES_FOR_GOF)
def test_ensemble_conditional_gof(N, V, C, M):
    ensemble = generate_fake_ensemble(N, V, C, M)
    server = EnsembleServer(ensemble)
    validate_gof(N, V, C, M, server, conditional=True)


def validate_gof(N, V, C, M, server, conditional):
    # Generate samples.
    expected = C**V
    num_samples = 1000 * expected
    ones = np.ones(V, dtype=np.int8)
    if conditional:
        cond_data = server.sample(1, ones)[0, :]
    else:
        cond_data = server.make_zero_row()
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

    # Truncate to avoid low-precision.
    truncated = False
    valid = (probs * num_samples > 20)
    if not valid.all():
        T = valid.argmin()
        T = max(8, T)  # Avoid truncating too much
        probs = probs[:T]
        counts = counts[:T]
        truncated = True

    gof = multinomial_goodness_of_fit(
        probs, counts, num_samples, plot=True, truncated=truncated)
    assert 1e-2 < gof


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 2, 3, 3),
    (10, 3, 4, 4),
    (10, 4, 5, 5),
    (10, 4, 6, 6),
    (10, 4, 7, 7),
    (10, 4, 8, 8),
])
def test_observed_perplexity(N, V, C, M):
    set_random_seed(make_seed(N, V, C, M))
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)

    for count in [1, 2, 3]:
        if count > 1 and C > 2:
            continue  # NotImplementedError.
        counts = 1
        perplexity = server.observed_perplexity(counts)
        print(perplexity)
        assert perplexity.shape == (V, )
        assert np.all(1 <= perplexity)
        assert np.all(perplexity <= count * C)


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 2, 2, 3),
    (10, 3, 2, 4),
    (10, 4, 2, 5),
    (10, 5, 2, 6),
    (10, 6, 2, 7),
    (10, 7, 2, 8),
])
def test_latent_perplexity(N, V, C, M):
    set_random_seed(make_seed(N, V, C, M))
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)

    perplexity = server.latent_perplexity()
    print(perplexity)
    assert perplexity.shape == (V, )
    assert np.all(1 <= perplexity)
    assert np.all(perplexity <= M)


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 2, 2, 3),
    (10, 3, 2, 4),
    (10, 4, 2, 5),
    (10, 5, 2, 6),
    (10, 6, 2, 7),
    (10, 7, 2, 8),
])
def test_ensemble_latent_perplexity(N, V, C, M):
    set_random_seed(make_seed(N, V, C, M))
    ensemble = generate_fake_ensemble(N, V, C, M)
    server = EnsembleServer(ensemble)

    perplexity = server.latent_perplexity()
    print(perplexity)
    assert perplexity.shape == (V, )
    assert np.all(1 <= perplexity)
    assert np.all(perplexity <= M)


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 2, 2, 3),
    (10, 3, 2, 4),
    (10, 4, 2, 5),
    (10, 5, 2, 6),
    (10, 6, 2, 7),
    (10, 7, 2, 8),
])
def test_latent_correlation(N, V, C, M):
    set_random_seed(make_seed(N, V, C, M))
    model = generate_fake_model(N, V, C, M)
    config = TINY_CONFIG.copy()
    config['model_num_clusters'] = M
    model['config'] = config
    server = TreeCatServer(model)

    correlation = server.latent_correlation()
    print(correlation)
    assert np.all(0 <= correlation)
    assert np.all(correlation <= 1)
    assert np.allclose(correlation, correlation.T)
    for v in range(V):
        assert correlation[v, :].argmax() == v
        assert correlation[:, v].argmax() == v


@pytest.mark.parametrize('N,V,C,M', [
    (10, 1, 2, 2),
    (10, 2, 2, 3),
    (10, 3, 2, 4),
    (10, 4, 2, 5),
    (10, 5, 2, 6),
    (10, 6, 2, 7),
    (10, 7, 2, 8),
])
def test_ensemble_latent_correlation(N, V, C, M):
    set_random_seed(make_seed(N, V, C, M))
    ensemble = generate_fake_ensemble(N, V, C, M)
    server = EnsembleServer(ensemble)

    correlation = server.latent_correlation()
    print(correlation)
    assert np.all(0 <= correlation)
    assert np.all(correlation <= 1)
    assert np.allclose(correlation, correlation.T)
    for v in range(V):
        assert correlation[v, :].argmax() == v
        assert correlation[:, v].argmax() == v
