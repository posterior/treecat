from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.util import jit_sample_from_probs
from treecat.util import parallel_map
from treecat.util import quantize_from_probs2
from treecat.util import sample_from_probs2
from treecat.util import set_random_seed
from treecat.util import sizeof


@pytest.mark.parametrize('shape,dtype,expected_size', [
    ([], np.int8, 1),
    ([], np.int32, 4),
    ([], np.int64, 8),
    ([], np.float32, 4),
    ([], np.float64, 8),
    ([3], np.int32, 3 * 4),
    ([3, 5], np.int32, 3 * 5 * 4),
    ([3, 5, 7], np.int32, 3 * 5 * 7 * 4),
])
def test_sizeof_numpy(shape, dtype, expected_size):
    assert sizeof(np.zeros(shape, dtype)) == expected_size


@pytest.mark.parametrize('size', range(1, 15))
def test_jit_sample_from_probs_gof(size):
    set_random_seed(size)
    probs = np.exp(2 * np.random.random(size)).astype(np.float32)
    counts = np.zeros(size, dtype=np.int32)
    num_samples = 2000 * size
    for _ in range(num_samples):
        counts[jit_sample_from_probs(probs)] += 1
    probs /= probs.sum()  # Normalize afterwards.
    print(counts)
    print(probs * num_samples)
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof


@pytest.mark.parametrize('size', range(1, 15))
def test_sample_from_probs2_gof(size):
    set_random_seed(size)
    probs = np.exp(2 * np.random.random(size)).astype(np.float32)
    counts = np.zeros(size, dtype=np.int32)
    num_samples = 2000 * size
    probs2 = np.tile(probs, (num_samples, 1))
    samples = sample_from_probs2(probs2)
    probs /= probs.sum()  # Normalize afterwards.
    counts = np.bincount(samples, minlength=size)
    print(counts)
    print(probs * num_samples)
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof


@pytest.mark.parametrize('size,resolution', [
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
])
def test_quantize_from_probs2(size, resolution):
    set_random_seed(10 * size + resolution)
    probs = np.exp(np.random.random(size)).astype(np.float32)
    probs2 = probs.reshape((1, size))
    quantized = quantize_from_probs2(probs2, resolution)
    assert quantized.shape == probs2.shape
    assert quantized.dtype == np.int8
    assert np.all(quantized.sum(axis=1) == resolution)

    # Check that quantized result is closer to target than any other value.
    quantized = quantized.reshape((size, ))
    target = resolution * probs / probs.sum()
    distance = np.abs(quantized - target).sum()
    for combo in itertools.combinations(range(size), resolution):
        other = np.zeros(size, np.int8)
        for i in combo:
            other[i] += 1
        assert other.sum() == resolution
        other_distance = np.abs(other - target).sum()
        assert distance <= other_distance


def _simple_function(x):
    return x * (x + 1) // 2


@pytest.mark.parametrize('size', range(10))
def test_parallel_map(size):
    args = list(range(10))
    expected = list(map(_simple_function, args))
    actual = parallel_map(_simple_function, args)
    assert actual == expected
