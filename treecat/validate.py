from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
from parsable import parsable

from six.moves import range
from six.moves import zip
from treecat.config import make_config
from treecat.format import csv_reader
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.format import pickle_memoize
from treecat.serving import TreeCatServer
from treecat.training import train_model
from treecat.util import guess_counts
from treecat.util import make_ragged_mask
from treecat.util import parallel_map
from treecat.util import set_random_seed

parsable = parsable.Parsable()

Stats = namedtuple('Stats', ['logprob', 'l1_loss'])


def make_splits(ragged_index, num_rows, num_parts):
    """Split a dataset for unsupervised crossvalidation.

    This splits a dataset into num_parts disjoint parts by randomly holding out
    cells. Note that whereas supervised crossvalidation typically holds out
    entire rows, our unsupervised crossvalidation is intended to evaluate a
    model of the full joint distribution.

    Args:
        ragged_index: A [V+1]-shaped numpy array of indices into the ragged
            data array, where V is the number of features.
        num_rows: An integer, the number of rows in the dataset.
        num_parts: An integer, the number of folds in n-fold crossvalidation.

    Returns:
        A num_parts-long list of mostly-empty [N,R]-shaped masks, where
        N = num_rows, and R = ragged_index[-1].
    """
    N = num_rows
    V = ragged_index.shape[0] - 1
    R = ragged_index[-1]
    holdouts = np.random.randint(num_parts, size=(N, V))
    masks = []
    for i in range(num_parts):
        dense_mask = (holdouts == i)
        ragged_mask = make_ragged_mask(ragged_index, dense_mask.T).T
        assert ragged_mask.shape == (N, R)
        masks.append(ragged_mask)
    return masks


memoized_train_model = pickle_memoize(train_model)


def _crossvalidate(task):
    (key, ragged_index, counts, data, mask, config) = task
    part = data.copy()
    part[mask] = 0
    print('training {}'.format(key))
    model = memoized_train_model(ragged_index, part, config)
    server = TreeCatServer(model)
    print('evaluating {}'.format(key))
    logprob = np.mean(server.logprob(data) - server.logprob(part))
    median = server.median(counts, part)
    l1_loss = np.abs(median - data)[mask].sum()
    return key, Stats(logprob, l1_loss)


def plan_crossvalidation(key, ragged_index, data, config):
    set_random_seed(config['seed'])
    counts = guess_counts(ragged_index, data)
    num_rows = data.shape[0]
    num_parts = config['model_ensemble_size']
    masks = make_splits(ragged_index, num_rows, num_parts)
    tasks = []
    for sub_seed, mask in enumerate(masks):
        sub_config = config.copy()
        sub_config['seed'] += sub_seed
        tasks.append((key, ragged_index, counts, data, mask, sub_config))
    return tasks


@parsable
def tune_csv(dataset_path, param_csv_path, result_path, **options):
    """Tune parameters specified in a csv file."""
    # Read csv file of parameters.
    configs = {}
    with csv_reader(param_csv_path) as reader:
        header = next(reader)
        for row in reader:
            if len(row) != len(header) or row[0].startswith('#'):
                continue
            for key, value in zip(header, row):
                options[key] = int(value)
            configs[tuple(row)] = make_config(**options)

    # Run grid search.
    dataset = pickle_load(dataset_path)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    tasks = []
    for key, config in configs.items():
        tasks += plan_crossvalidation(key, ragged_index, data, config)
    print('tuning via {} tasks'.format(len(tasks)))
    result = parallel_map(_crossvalidate, tasks)
    for line in sorted(result):
        print(line)
    pickle_dump(result, result_path)


if __name__ == '__main__':
    parsable()
