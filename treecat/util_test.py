from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.util import jit_sample_from_probs
from treecat.util import parallel_map
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
    probs /= probs.sum()
    counts = np.zeros(size, dtype=np.int32)
    num_samples = 2000 * size
    for _ in range(num_samples):
        counts[jit_sample_from_probs(probs)] += 1
    print(counts)
    print(probs * num_samples)
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof


@pytest.mark.parametrize('size', range(1, 15))
def test_sample_from_probs2_gof(size):
    set_random_seed(size)
    probs = np.exp(2 * np.random.random(size)).astype(np.float32)
    probs /= probs.sum()
    counts = np.zeros(size, dtype=np.int32)
    num_samples = 2000 * size
    probs2 = np.tile(probs, (num_samples, 1))
    samples = sample_from_probs2(probs2)
    counts = np.bincount(samples, minlength=size)
    print(counts)
    print(probs * num_samples)
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof


def _simple_function(x):
    return x * (x + 1) // 2


@pytest.mark.parametrize('size', range(10))
def test_parallel_map(size):
    args = list(range(10))
    expected = list(map(_simple_function, args))
    actual = parallel_map(_simple_function, args)
    assert actual == expected
