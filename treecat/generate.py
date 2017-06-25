from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from parsable import parsable
from treecat.config import DEFAULT_CONFIG
from treecat.persist import pickle_dump
from treecat.persist import pickle_load
from treecat.training import train_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'generated')


def generate_dataset(num_rows, num_cols, num_cats, density=0.9):
    """Generate a random dataset.

    Returns:
      The path to a gzipped pickled (data, mask) pair.
    """
    path = os.path.join(DATA, '{}-{}-{}-{:0.1f}.dataset.pkl.gz'.format(
        num_rows, num_cols, num_cats, density))
    if os.path.exists(path):
        return path
    print('Generating {}'.format(path))
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    np.random.seed(0)
    shape = [num_rows, num_cols]
    data = np.random.randint(num_cats, size=shape, dtype=np.int32)
    mask = np.random.random(size=shape) < density
    pickle_dump((data, mask), path)
    return path


def generate_model(num_rows, num_cols, num_cats, density=0.9):
    """Generate a random model.

    Returns:
      The path to a gzipped pickled model.
    """
    path = os.path.join(DATA, '{}-{}-{}-{:0.1f}.model.pkl.gz'.format(
        num_rows, num_cols, num_cats, density))
    if os.path.exists(path):
        return path
    print('Generating {}'.format(path))
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    dataset_path = generate_dataset(num_rows, num_cols, num_cats, density)
    data, mask = pickle_load(dataset_path)
    config = DEFAULT_CONFIG.copy()
    config['annealing_epochs'] = 5
    model = train_model(data, mask, config)
    pickle_dump(model, path)
    return path


@parsable
def clear():
    """Clear cache of generated datasets."""
    if os.path.exists(DATA):
        shutil.rmtree(DATA)


if __name__ == '__main__':
    parsable()
