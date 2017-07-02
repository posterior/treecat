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
from treecat.structure import TreeStructure
from treecat.structure import sample_tree
from treecat.training import train_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'generated')


def generate_dataset(num_rows, num_cols, num_cats=4, rate=1.0):
    """Generate a random dataset.

    Returns:
      A [num_cols] list of [num_rows, num_cats] multinomial columns.
    """
    np.random.seed(0)
    data = [None] * num_cols
    for v in range(num_cols):
        probs = np.random.dirichlet(np.zeros(num_cats) + 0.5)
        data[v] = np.zeros((num_rows, num_cats), dtype=np.int8)
        for n in range(num_rows):
            count = np.random.poisson(rate)
            data[v][n, :] = np.random.multinomial(count, probs)
    return data


def generate_dataset_file(num_rows, num_cols, num_cats=4, rate=1.0):
    """Generate a random dataset.

    Returns:
      The path to a gzipped pickled data table.
    """
    path = os.path.join(DATA, '{}-{}-{}-{:0.1f}.dataset.pkl.gz'.format(
        num_rows, num_cols, num_cats, rate))
    if os.path.exists(path):
        return path
    print('Generating {}'.format(path))
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    data = generate_dataset(num_rows, num_cols, num_cats, rate)
    pickle_dump(data, path)
    return path


def generate_tree(num_cols):
    tree = TreeStructure(num_cols)
    K = tree.complete_grid.shape[1]
    edge_logits = np.random.random([K])
    edges = [tuple(edge) for edge in tree.tree_grid[1:3, :].T]
    edges = sample_tree(tree.complete_grid, edge_logits, edges, steps=10)
    tree.set_edges(edges)
    return tree


def generate_fake_model(num_rows, num_cols, num_cats=4, rate=1.0):
    tree = generate_tree(num_cols)
    raise NotImplementedError()
    return {'tree': tree, 'suffstats': {}}


def generate_model_file(num_rows, num_cols, num_cats=4, rate=1.0):
    """Generate a random model.

    Returns:
      The path to a gzipped pickled model.
    """
    path = os.path.join(DATA, '{}-{}-{}-{:0.1f}.model.pkl.gz'.format(
        num_rows, num_cols, num_cats, rate))
    if os.path.exists(path):
        return path
    print('Generating {}'.format(path))
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    dataset_path = generate_dataset_file(num_rows, num_cols, num_cats, rate)
    data = pickle_load(dataset_path)
    config = DEFAULT_CONFIG.copy()
    config['learning_annealing_epochs'] = 5
    model = train_model(data, config)
    pickle_dump(model, path)
    return path


@parsable
def clean():
    """Clean out cache of generated datasets."""
    if os.path.exists(DATA):
        shutil.rmtree(DATA)


if __name__ == '__main__':
    parsable()
