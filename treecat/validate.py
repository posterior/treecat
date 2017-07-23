from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from parsable import parsable

from six.moves import range
from six.moves import zip
from treecat.config import make_config
from treecat.format import csv_reader
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.serving import TreeCatServer
from treecat.training import train_model
from treecat.util import count_observations
from treecat.util import get_profiling_stats
from treecat.util import make_ragged_mask
from treecat.util import parallel_map
from treecat.util import profile
from treecat.util import set_random_seed

parsable = parsable.Parsable()


def serialize_config(config):
    """Serialize a config dict to a short string for use in filenames."""
    keys = sorted(config.keys())
    assert keys == sorted(make_config().keys())
    return '-'.join(str(int(config[key])) for key in keys)


def split_data(ragged_index, num_rows, num_parts, partid):
    """Split a dataset into training + holdout for n-fold crossvalidation.

    This splits a dataset into num_parts disjoint parts by randomly holding out
    cells. Note that whereas supervised crossvalidation typically holds out
    entire rows, our unsupervised crossvalidation is intended to evaluate a
    model of the full joint distribution.

    Args:
        ragged_index: A [V+1]-shaped numpy array of indices into the ragged
            data array, where V is the number of features.
        num_rows: An integer, the number of rows in the dataset.
        num_parts: An integer, the number of folds in n-fold crossvalidation.
        partid: An integer in [0, num_parts).

    Returns:
        A [N,R]-shaped mask where True means held-out and False means training.
        Here N = num_rows and R = ragged_index[-1].
    """
    set_random_seed(0)
    assert 0 <= partid < num_parts
    N = num_rows
    V = ragged_index.shape[0] - 1
    R = ragged_index[-1]
    dense_mask = (partid == np.random.randint(num_parts, size=(N, V)))
    ragged_mask = make_ragged_mask(ragged_index, dense_mask.T).T
    assert ragged_mask.shape == (N, R)
    return ragged_mask


def read_param_csv(param_csv_path, **options):
    """Reads configs from a csv file.

    Args:
      param_csv_path: The path to a csv file with one line per config.
      options: A dict of extra config parameters.

    Returns:
      A pair (header, configs), where:
      header is a list of parameters, and
      configs is list of config dicts.
    """
    # This is a hack: to configure crossvalidation from a config, we alias:
    #   num_parts = model_ensemble_size
    #   partid = seed
    num_parts = options.get('model_ensemble_size', 4)
    assert 'seed' not in options
    configs = []
    with csv_reader(param_csv_path) as reader:
        header = next(reader)
        assert 'seed' not in header
        assert 'model_ensemble_size' not in header
        for row in reader:
            if len(row) != len(header) or row[0].startswith('#'):
                continue
            for key, value in zip(header, row):
                options[key] = int(value)
            for partid in range(num_parts):
                configs.append(make_config(seed=partid, **options))
    return header, configs


def process_train_task(task):
    (dataset_path, config, models_dir) = task
    model_path = os.path.join(models_dir,
                              'model.{}.pkz'.format(serialize_config(config)))
    if os.path.exists(model_path):
        return
    print('Train {}'.format(os.path.basename(model_path)))

    # Split data for crossvalidation.
    num_parts = config['model_ensemble_size']
    partid = config['seed']
    assert 0 <= partid < num_parts
    dataset = pickle_load(dataset_path)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    num_rows = data.shape[0]
    mask = split_data(ragged_index, num_rows, num_parts, partid)
    training_data = data
    training_data[mask] = 0

    # Train a model.
    model = train_model(ragged_index, training_data, config)
    model['profiling_stats'] = get_profiling_stats()
    pickle_dump(model, model_path)


@parsable
def train(dataset_path, param_csv_path, models_dir, **options):
    """Tune parameters specified in a csv file."""
    options = {k: int(v) for k, v in options.items()}
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    header, configs = read_param_csv(param_csv_path, **options)
    configs.sort(key=lambda c: c['learning_init_epochs'])
    tasks = [(dataset_path, c, models_dir) for c in configs]
    print('Scheduling {} tasks'.format(len(tasks)))
    parallel_map(process_train_task, tasks)


@profile
def process_eval_task(task):
    (dataset_path, config, models_dir) = task

    # Load a server with the trained model.
    model_path = os.path.join(models_dir,
                              'model.{}.pkz'.format(serialize_config(config)))
    try:
        model = pickle_load(model_path)
    except (OSError, EOFError):
        return {'config': config}
    print('Eval {}'.format(os.path.basename(model_path)))
    server = TreeCatServer(model)

    # Split data for crossvalidation.
    num_parts = config['model_ensemble_size']
    partid = config['seed']
    assert 0 <= partid < num_parts
    dataset = pickle_load(dataset_path)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    num_rows = data.shape[0]
    mask = split_data(ragged_index, num_rows, num_parts, partid)
    training_data = data.copy()
    training_data[mask] = 0
    validation_data = data.copy()
    validation_data[~mask] = 0

    # Compute posterior predictive log probability of held-out data.
    logprob = np.mean(server.logprob(data) - server.logprob(training_data))

    # Compute L1 loss on observed validation data.
    N, R = data.shape
    V = len(ragged_index) - 1
    obs_counts = count_observations(ragged_index, data)
    assert obs_counts.shape == (N, V)
    max_counts = obs_counts.max(axis=0)
    median = server.median(max_counts, training_data)
    observed = (obs_counts == max_counts[np.newaxis, :])
    observed = make_ragged_mask(ragged_index, observed.T).T
    relevant = observed & mask
    validation_data[~relevant] = 0
    median[~relevant] = 0
    l1_loss = 0.5 * np.abs(median - validation_data).sum()
    l1_loss /= relevant.sum() + 0.1

    return {'config': config, 'logprob': logprob, 'l1_loss': l1_loss}


@parsable
def eval(dataset_path, param_csv_path, models_dir, result_path, **options):
    """Evaluate trained models."""
    options = {k: int(v) for k, v in options.items()}
    header, configs = read_param_csv(param_csv_path, **options)
    tasks = [(dataset_path, c, models_dir) for c in configs]
    print('Scheduling {} tasks'.format(len(tasks)))
    result = parallel_map(process_eval_task, tasks)

    print('\t'.join(header + ['pogprob', 'l1_loss']))
    lines = [
        [row['config'][param] for param in header] +  #
        [row['logprob'], row['l1_loss']]  #
        for row in result if 'logprob' in row and 'l1_loss' in row
    ]
    lines.sort()
    for line in lines:
        print(' '.join('{:0.1f}'.format(cell) for cell in line))

    pickle_dump(result, result_path)


if __name__ == '__main__':
    parsable()
