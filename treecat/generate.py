from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
from parsable import parsable

from treecat.config import make_config
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.serving import TreeCatServer
from treecat.structure import TreeStructure
from treecat.structure import sample_tree
from treecat.training import train_model
from treecat.util import set_random_seed

parsable = parsable.Parsable()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'generated')


def generate_dataset(num_rows, num_cols, num_cats=4, rate=1.0):
    """Generate a random dataset.

    Returns:
      A pair (ragged_index, data).
    """
    set_random_seed(0)
    ragged_index = np.arange(0, num_cats * (num_cols + 1), num_cats, np.int32)
    data = np.zeros((num_rows, num_cols * num_cats), np.int8)
    for v in range(num_cols):
        beg, end = ragged_index[v:v + 2]
        column = data[:, beg:end]
        probs = np.random.dirichlet(np.zeros(num_cats) + 0.5)
        for n in range(num_rows):
            count = np.random.poisson(rate)
            column[n, :] = np.random.multinomial(count, probs)
    dataset = {
        'schema': {
            'ragged_index': ragged_index,
        },
        'data': data,
    }
    return dataset


def generate_dataset_file(num_rows, num_cols, num_cats=4, rate=1.0):
    """Generate a random dataset.

    Returns:
      The path to a gzipped pickled data table.
    """
    path = os.path.join(DATA, '{}-{}-{}-{:0.1f}.dataset.pkz'.format(
        num_rows, num_cols, num_cats, rate))
    if os.path.exists(path):
        return path
    print('Generating {}'.format(path))
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    dataset = generate_dataset(num_rows, num_cols, num_cats, rate)
    pickle_dump(dataset, path)
    return path


def generate_tree(num_cols):
    tree = TreeStructure(num_cols)
    K = tree.complete_grid.shape[1]
    edge_logits = np.random.random([K])
    edges = [tuple(edge) for edge in tree.tree_grid[1:3, :].T]
    edges = sample_tree(tree.complete_grid, edge_logits, edges, steps=10)
    tree.set_edges(edges)
    return tree


def generate_clean_dataset(tree, num_rows, num_cats):
    """Generate a dataset whose structure should be easy to learn.

    This generates a highly correlated uniformly distributed dataset with
    given tree structure. This is useful to test that structure learning can
    recover a known structure.

    Args:
      tree: A TreeStructure instance.
      num_rows: The number of rows in the generated dataset.
      num_cats: The number of categories in the geneated categorical dataset.
        This will also be used for the number of latent classes.

    Returns:
      A dict with keys 'schema' and 'data'. The schema will only have a
      'ragged' index field.
    """
    assert isinstance(tree, TreeStructure)
    V = tree.num_vertices
    E = V - 1
    K = V * (V - 1) // 2
    C = num_cats
    M = num_cats
    config = make_config(model_num_clusters=M)
    ragged_index = np.arange(0, C * (V + 1), C, np.int32)

    # Create sufficient statistics that are ideal for structure learning:
    # Correlation should be high enough that (vertex,vertex) correlation can be
    # detected, but low enough that multi-hop correlation can be distinguished
    # from single-hop correlation.
    # Observations should have very low error rate.
    edge_precision = 1
    feat_precision = 100
    vert_ss = np.zeros((V, M), dtype=np.int32)
    edge_ss = np.zeros((E, M, M), dtype=np.int32)
    feat_ss = np.zeros((V * C, M), dtype=np.int32)
    meas_ss = np.zeros([V, M], np.int32)
    vert_ss[...] = edge_precision
    meas_ss[...] = feat_precision
    for e, v1, v2 in tree.tree_grid.T:
        edge_ss[e, :, :] = edge_precision * np.eye(M, dtype=np.int32)
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        feat_ss[beg:end, :] = feat_precision * np.eye(M, dtype=np.int32)
    model = {
        'config': config,
        'tree': tree,
        'edge_logits': np.zeros(K, np.float32),
        'suffstats': {
            'ragged_index': ragged_index,
            'vert_ss': vert_ss,
            'edge_ss': edge_ss,
            'feat_ss': feat_ss,
            'meas_ss': meas_ss,
        },
    }
    server = TreeCatServer(model)
    data = server.sample(num_rows, counts=np.ones(V, np.int8))
    dataset = {
        'schema': {
            'ragged_index': ragged_index,
        },
        'data': data,
    }
    return dataset


def generate_fake_model(num_rows,
                        num_cols,
                        num_cats,
                        num_components,
                        dataset=None):
    tree = generate_tree(num_cols)
    assignments = np.random.choice(num_components, size=(num_rows, num_cols))
    assignments = assignments.astype(np.int32)
    if dataset is None:
        dataset = generate_dataset(num_rows, num_cols, num_cats)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    N = num_rows
    V = num_cols
    E = V - 1
    K = V * (V - 1) // 2
    C = num_cats
    M = num_components
    vert_ss = np.zeros((V, M), dtype=np.int32)
    edge_ss = np.zeros((E, M, M), dtype=np.int32)
    feat_ss = np.zeros((V * C, M), dtype=np.int32)
    meas_ss = np.zeros([V, M], np.int32)
    for v in range(V):
        vert_ss[v, :] = np.bincount(assignments[:, v], minlength=M)
    for e, v1, v2 in tree.tree_grid.T:
        pairs = assignments[:, v1].astype(np.int32) * M + assignments[:, v2]
        edge_ss[e, :, :] = np.bincount(pairs, minlength=M * M).reshape((M, M))
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        data_block = data[:, beg:end]
        feat_ss_block = feat_ss[beg:end, :]
        for n in range(N):
            feat_ss_block[:, assignments[n, v]] += data_block[n, :]
            meas_ss[v, assignments[n, v]] += data_block[n, :].sum()
    edge_logits = np.exp(-np.random.rand(K))
    model = {
        'tree': tree,
        'assignments': assignments,
        'edge_logits': edge_logits,
        'suffstats': {
            'ragged_index': ragged_index,
            'vert_ss': vert_ss,
            'edge_ss': edge_ss,
            'feat_ss': feat_ss,
            'meas_ss': meas_ss,
        },
    }
    return model


def generate_fake_ensemble(num_rows, num_cols, num_cats, num_components):
    dataset = generate_dataset(num_rows, num_cols, num_cats)
    ensemble = []
    config = make_config(model_num_clusters=num_components, seed=0)
    for sub_seed in range(3):
        sub_config = config.copy()
        sub_config['seed'] += sub_seed
        set_random_seed(sub_config['seed'])
        model = generate_fake_model(num_rows, num_cols, num_cats,
                                    num_components, dataset)
        model['config'] = sub_config
        ensemble.append(model)
    return ensemble


def generate_model_file(num_rows, num_cols, num_cats=4, rate=1.0):
    """Generate a random model.

    Returns:
      The path to a gzipped pickled model.
    """
    path = os.path.join(DATA, '{}-{}-{}-{:0.1f}.model.pkz'.format(
        num_rows, num_cols, num_cats, rate))
    if os.path.exists(path):
        return path
    print('Generating {}'.format(path))
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    dataset_path = generate_dataset_file(num_rows, num_cols, num_cats, rate)
    dataset = pickle_load(dataset_path)
    schema = dataset['schema']
    config = make_config(learning_init_epochs=5)
    model = train_model(schema['ragged_index'], dataset['data'], config)
    pickle_dump(model, path)
    return path


@parsable
def clean():
    """Clean out cache of generated datasets."""
    if os.path.exists(DATA):
        shutil.rmtree(DATA)


if __name__ == '__main__':
    parsable()
