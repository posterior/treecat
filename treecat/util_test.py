from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

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


@pytest.mark.parametrize('shape,dtype,expected_size', [
    ([], tf.int8, 1),
    ([], tf.int32, 4),
    ([], tf.int64, 8),
    ([], tf.float32, 4),
    ([], tf.float64, 8),
    ([3], tf.int32, 3 * 4),
    ([3, 5], tf.int32, 3 * 5 * 4),
    ([3, 5, 7], tf.int32, 3 * 5 * 7 * 4),
])
def test_sizeof_tensorflow(shape, dtype, expected_size):
    assert sizeof(tf.zeros(shape, dtype)) == expected_size
