from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
from parsable import parsable

from treecat.config import make_config
from treecat.format import csv_reader
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.serving import TreeCatServer
from treecat.training import train_model
from treecat.util import parallel_map

parsable = parsable.Parsable()

Stats = namedtuple('Stats', ['logprob', 'l1_loss'])


def split_data(ragged_index, data, num_parts):
    """Split a dataset for unsupervised crossvalidation.

    This splits a dataset into num_parts disjoint parts by randomly holding out
    cells. Note that whereas supervised crossvalidation typically holds out
    entire rows, our unsupervised crossvalidation is intended to evaluate a
    model of the full joint distribution.

    Args:
        ragged_index: A [V+1]-shaped numpy array of indices into the ragged
            data array, where V is the number of features.
        data: A [N, R]-shaped ragged nummpy array of multinomial count data,
            where N is the number of rows, and R is ragged_index[-1].
        num_parts: An integer, the number of folds in n-fold crossvalidation.

    Returns:
        A num_parts-long list of arrays shaped like data, each with some of its
        cells zeroed out.
    """
    N = data.shape[0]
    V = ragged_index.shape[0] - 1
    holdouts = np.random.randint(num_parts, size=(N, V))
    parts = []
    for i in range(num_parts):
        holdout = (holdouts == i)
        part = data.copy()
        for v in range(V):
            beg, end = ragged_index[v:v + 2]
            part[holdout[:, v], beg:end] = 0
        parts.append(part)
    return parts


def guess_counts(ragged_index, data):
    """Guess the multinomial count of each feature.

    This should guess 1 for categoricals and max-min for ordinals.
    """
    V = len(ragged_index) - 1
    counts = np.zeros(V, np.int8)
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        counts[v] = data[:, beg:end].sum(axis=1).max()
    return counts


def _crossvalidate(task):
    (key, ragged_index, counts, data, part, config) = task
    print('training {}'.format(key))
    model = train_model(ragged_index, data, config)
    server = TreeCatServer(model)
    print('evaluating {}'.format(key))
    logprob = np.mean(server.logprob(data) - server.logprob(part))
    # FIXME This should restrict to the held-out portion of the median.
    l1_loss = np.abs(server.median(counts, part) - data).sum()
    return key, Stats(logprob, l1_loss)


def plan_crossvalidation(key, ragged_index, data, config):
    counts = guess_counts(ragged_index, data)
    num_parts = config['model_ensemble_size']
    parts = split_data(ragged_index, data, num_parts)
    tasks = []
    for sub_seed, part in enumerate(parts):
        sub_config = config.copy()
        sub_config['seed'] += sub_seed
        tasks.append((key, ragged_index, counts, data, part, sub_config))
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
            row = tuple(map(int, row))
            for key, value in zip(header, row):
                options[key] = int(value)
            configs[row] = make_config(**options)

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