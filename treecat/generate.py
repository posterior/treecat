from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from treecat.config import DEFAULT_CONFIG


def generate_dataset(num_rows, num_cols, density=0.9, config=None):
    if config is None:
        config = DEFAULT_CONFIG
    np.random.seed(config['seed'])
    shape = [num_rows, num_cols]
    data = np.random.randint(
        config['num_categories'], size=shape, dtype=np.int32)
    mask = np.random.random(size=shape) < density
    return data, mask
