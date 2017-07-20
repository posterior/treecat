from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from goftests import multinomial_goodness_of_fit

from treecat.config import make_config
from treecat.generate import generate_clean_dataset
from treecat.generate import generate_dataset
from treecat.generate import generate_tree
from treecat.structure import TreeStructure
from treecat.structure import estimate_tree
from treecat.structure import print_tree
from treecat.structure import triangular_to_square
from treecat.testutil import numpy_seterr
from treecat.training import TreeCatTrainer
from treecat.training import get_annealing_schedule
from treecat.training import train_ensemble
from treecat.training import train_model
from treecat.util import np_printoptions
from treecat.util import set_random_seed

numpy_seterr()


def test_get_annealing_schedule():
    set_random_seed(0)
    num_rows = 10
    init_epochs = 10
    schedule = get_annealing_schedule(num_rows, init_epochs)
    for step, (action, row_id) in enumerate(schedule):
        assert step < 1000
        assert action in ['add_row', 'remove_row', 'sample_tree']
        if action == 'sample_tree':
            assert row_id is None
        else:
            assert 0 <= row_id and row_id < num_rows


def validate_model(ragged_index, data, model, config):
    assert model['config'] == config
    assert isinstance(model['tree'], TreeStructure)
    assert np.all(model['suffstats']['ragged_index'] == ragged_index)
    grid = model['tree'].tree_grid
    assignments = model['assignments']
    vert_ss = model['suffstats']['vert_ss']
    edge_ss = model['suffstats']['edge_ss']
    feat_ss = model['suffstats']['feat_ss']
    meas_ss = model['suffstats']['meas_ss']

    # Check shape.
    V = len(ragged_index) - 1
    N = data.shape[0]
    E = V - 1
    M = config['model_num_clusters']
    assert grid.shape == (3, E)
    assert assignments.shape == (N, V)
    assert vert_ss.shape == (V, M)
    assert edge_ss.shape == (E, M, M)
    assert feat_ss.shape == (ragged_index[-1], M)
    assert meas_ss.shape == (V, M)

    # Check bounds.
    assert np.all(0 <= assignments)
    assert np.all(assignments < M)
    assert np.all(0 <= vert_ss)
    assert np.all(vert_ss <= N)
    assert np.all(0 <= edge_ss)
    assert np.all(edge_ss <= N)
    assert np.all(0 <= feat_ss)
    assert np.all(0 <= meas_ss)

    # Check marginals.
    assert vert_ss.sum() == N * V
    assert np.all(vert_ss.sum(1) == N)
    assert edge_ss.sum() == N * E
    assert np.all(edge_ss.sum((1, 2)) == N)
    assert np.all(edge_ss.sum(2) == vert_ss[grid[1, :]])
    assert np.all(edge_ss.sum(1) == vert_ss[grid[2, :]])
    assert feat_ss.sum() == meas_ss.sum()
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        data_block = data[:, beg:end]
        feat_ss_block = feat_ss[beg:end, :]
        assert feat_ss_block.sum() == data_block.sum()
        assert np.all(feat_ss_block.sum(1) == data_block.sum(0))
        assert np.all(feat_ss_block.sum(0) == meas_ss[v, :])

    # Check computation from scratch.
    for v in range(V):
        counts = np.bincount(assignments[:, v], minlength=M)
        assert np.all(vert_ss[v, :] == counts)
    for e, v1, v2 in grid.T:
        pairs = assignments[:, v1].astype(np.int32) * M + assignments[:, v2]
        counts = np.bincount(pairs, minlength=M * M).reshape((M, M))
        assert np.all(edge_ss[e, :, :] == counts)
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        feat_ss_block = feat_ss[beg:end, :]
        counts = np.zeros_like(feat_ss_block)
        for n in range(N):
            counts[:, assignments[n, v]] += data[n, beg:end]
        assert np.all(feat_ss_block == counts)
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        counts = np.zeros_like(meas_ss[v, :])
        for n in range(N):
            counts[assignments[n, v]] += data[n, beg:end].sum()
        assert np.all(meas_ss[v, :] == counts)


@pytest.mark.parametrize('N,V,C,M', [
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (3, 3, 3, 3),
    (4, 4, 4, 4),
    (5, 5, 5, 5),
    (6, 6, 6, 6),
])
def test_train_model(N, V, C, M):
    config = make_config()
    config['model_num_clusters'] = M
    dataset = generate_dataset(num_rows=N, num_cols=V, num_cats=C)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    model = train_model(ragged_index, data, config)
    validate_model(ragged_index, data, model, config)


@pytest.mark.parametrize('N,V,C,M', [
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (3, 3, 3, 3),
    (4, 4, 4, 4),
    (5, 5, 5, 5),
    (6, 6, 6, 6),
])
def test_train_ensemble(N, V, C, M):
    config = make_config()
    config['model_num_clusters'] = M
    dataset = generate_dataset(num_rows=N, num_cols=V, num_cats=C)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    ensemble = train_ensemble(ragged_index, data, config)

    assert len(ensemble) == config['model_ensemble_size']
    for sub_seed, model in enumerate(ensemble):
        sub_config = config.copy()
        sub_config['seed'] += sub_seed
        validate_model(ragged_index, data, model, sub_config)


def hash_assignments(assignments):
    assert isinstance(assignments, np.ndarray)
    return tuple(tuple(row) for row in assignments)


@pytest.mark.parametrize('N,V,C,M', [
    (1, 1, 1, 1),
    (1, 2, 2, 2),
    (1, 2, 2, 3),
    (1, 3, 2, 2),
    (1, 4, 2, 2),
    (2, 1, 2, 2),
    (2, 1, 2, 3),
    (2, 2, 2, 2),
    (2, 2, 2, 3),
    (2, 2, 3, 2),
    (2, 2, 4, 2),
    (2, 3, 2, 2),
    (3, 1, 2, 2),
    (3, 2, 2, 2),
    (4, 1, 2, 2),
])
def test_assignment_sampler_gof(N, V, C, M):
    config = make_config()
    config['model_num_clusters'] = M
    dataset = generate_dataset(num_rows=N, num_cols=V, num_cats=C)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    trainer = TreeCatTrainer(ragged_index, data, config)
    print('Data:')
    print(data)

    # Add all rows.
    set_random_seed(1)
    for row_id in range(N):
        trainer.add_row(row_id)

    # Collect samples.
    num_samples = 500 * M**(N * V)
    counts = {}
    logprobs = {}
    for _ in range(num_samples):
        for row_id in range(N):
            # This is a single-site Gibbs sampler.
            trainer.remove_row(row_id)
            trainer.add_row(row_id)
        key = hash_assignments(trainer._assignments)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
            logprobs[key] = trainer.logprob()
    assert len(counts) == M**(N * V)

    # Check accuracy using Pearson's chi-squared test.
    keys = sorted(counts.keys())
    counts = np.array([counts[k] for k in keys], dtype=np.int32)
    probs = np.exp(np.array([logprobs[k] for k in keys]))
    probs /= probs.sum()
    print('Actual\tExpected\tAssignment')
    for count, prob, key in zip(counts, probs, keys):
        print('{:}\t{:0.1f}\t{}'.format(count, prob * num_samples, key))
    gof = multinomial_goodness_of_fit(probs, counts, num_samples, plot=True)
    assert 1e-2 < gof


@pytest.mark.parametrize('V,C', [
    (1, 2),
    (2, 2),
    (2, 3),
    (3, 2),
    pytest.mark.xfail((3, 3)),
    (4, 3),
    (5, 3),
    pytest.mark.xfail((6, 3)),
])
def test_recover_structure(V, C):
    set_random_seed(V + C * 10)
    N = 200
    M = 2 * C
    tree = generate_tree(num_cols=V)
    dataset = generate_clean_dataset(tree, num_rows=N, num_cats=C)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    config = make_config(model_num_clusters=M)
    model = train_model(ragged_index, data, config)

    # Compute three types of edges.
    expected_edges = tree.get_edges()
    optimal_edges = estimate_tree(tree.complete_grid, model['edge_logits'])
    actual_edges = model['tree'].get_edges()

    # Print debugging information.
    feature_names = [str(v) for v in range(V)]
    root = '0'
    readable_data = np.zeros([N, V], np.int8)
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        readable_data[:, v] = data[:, beg:end].argmax(axis=1)
    with np_printoptions(precision=2, threshold=100, edgeitems=5):
        print('Expected:')
        print(print_tree(expected_edges, feature_names, root))
        print('Optimal:')
        print(print_tree(optimal_edges, feature_names, root))
        print('Actual:')
        print(print_tree(actual_edges, feature_names, root))
        print('Correlation:')
        print(np.corrcoef(readable_data.T))
        print('Edge logits:')
        print(triangular_to_square(tree.complete_grid, model['edge_logits']))
        print('Data:')
        print(readable_data)
        print('Feature Sufficient Statistics:')
        print(model['suffstats']['feat_ss'])
        print('Edge Sufficient Statistics:')
        print(model['suffstats']['edge_ss'])

    # Check agreement.
    assert actual_edges == optimal_edges, 'Error in sample_tree'
    assert actual_edges == expected_edges, 'Error in likelihood'
