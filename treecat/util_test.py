from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.training import sample_from_probs
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


@pytest.mark.parametrize('size', range(1, 10))
def test_sample_from_probs(size):
    np.random.seed(size)
    probs = np.exp(2 * np.random.random(size)).astype(np.float32)
    probs /= probs.sum()
    counts = np.zeros(size, dtype=np.int32)
    num_samples = 2000 * size
    for _ in range(num_samples):
        counts[sample_from_probs(probs)] += 1
    print(counts)
    print(probs * num_samples)
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof
