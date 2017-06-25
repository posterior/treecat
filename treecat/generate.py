from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys

import numpy as np

from parsable import parsable
from treecat.persist import pickle_dump

DATA = os.path.join(os.environ.get('HOME', '/tmp'), 'treecat', 'generated')


def generate_dataset(num_rows, num_cols, num_cats, density=0.9):
    np.random.seed(0)
    shape = [num_rows, num_cols]
    data = np.random.randint(num_cats, size=shape, dtype=np.int32)
    mask = np.random.random(size=shape) < density
    return data, mask


def generate_dataset_file(num_rows, num_cols, num_cats, density=0.9):
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    path = os.path.join(DATA, '{}-{}-{}-{:0.1f}.pkl.gz'.format(
        num_rows, num_cols, num_cats, density))
    if not os.path.exists(path):
        sys.stderr.write('Generating {}\n'.format(path))
        sys.stderr.flush()
        data, mask = generate_dataset(num_rows, num_cols, num_cats, density)
        pickle_dump((data, mask), path)
    return path


@parsable
def clear():
    """Clear cache of generated datasets."""
    if os.path.exists(DATA):
        shutil.rmtree(DATA)


if __name__ == '__main__':
    parsable()
