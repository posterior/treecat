from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

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
