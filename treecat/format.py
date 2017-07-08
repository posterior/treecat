from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import io
import logging
import re
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from parsable import parsable

from six.moves import cPickle as pickle
from six.moves import intern
from six.moves import zip

logger = logging.getLogger(__name__)
parsable = parsable.Parsable()

CATEGORICAL = intern('categorical')
ORDINAL = intern('ordinal')
VALID_TYPES = (CATEGORICAL, ORDINAL)
MAX_ORDINAL = 20
NA_STRINGS = frozenset([intern(''), intern('null'), intern('none')])


def pickle_dump(data, filename):
    """Pickle data to file using gzip compression."""
    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(filename):
    """Unpickle data from file using gzip compression."""
    with gzip.GzipFile(filename, 'rb') as f:
        return pickle.load(f)


@contextmanager
def csv_reader(filename, encoding='utf-8'):
    with io.open(filename, 'r', encoding=encoding) as f:
        yield csv.reader(f)


@contextmanager
def csv_writer(filename):
    with open(filename, 'w') as f:
        yield csv.writer(f)


def normalize_string(string):
    return re.sub(r'\s+', ' ', string.strip().lower())


def is_small_int(value):
    try:
        int_value = int(value)
        return 0 <= int_value and int_value <= MAX_ORDINAL
    except ValueError:
        return False


def guess_feature_type(count, values):
    """Guess the type of a feature, given statistics about the feature.

    Args:
      count: Total number of observations of the feature.
      values: A list of uniqe observed values of the feature.

    Returns:
      One of: 'ordinal', 'categorical', or ''
    """
    if len(values) <= 1:
        return ''  # Feature is useless.
    if len(values) <= 1 + MAX_ORDINAL and all(map(is_small_int, values)):
        return ORDINAL
    if len(values) <= min(count / 2, MAX_ORDINAL):
        return CATEGORICAL
    return ''


@parsable
def guess_schema(data_csv_in, schema_csv_out, encoding='utf-8'):
    """Create a best-guess type schema for a given dataset.

    Common encodings include: utf-8, cp1252.
    """
    print('Guessing schema of {}'.format(data_csv_in))

    # Collect statistics.
    totals = Counter()
    values = defaultdict(Counter)
    with csv_reader(data_csv_in, encoding) as reader:
        feature_names = [intern(n) for n in next(reader)]
        for row in reader:
            for name, value in zip(feature_names, row):
                value = normalize_string(value)
                if value in NA_STRINGS:
                    continue
                totals[name] += 1
                values[name][value] += 1

    # Exclude singleton values, because they provide no statistical value,
    # and they often leak identifying info.
    for name in feature_names:
        counts = values[name]
        singletons = [v for v, c in counts.items() if c == 1]
        for value in singletons:
            del counts[value]
            totals[name] -= 1
        values[name] = [v for (v, c) in counts.most_common(1 + MAX_ORDINAL)]
        values[name].sort(key=lambda v: (-counts[v], v))  # Brake ties.

    # Guess feature types.
    feature_types = [
        guess_feature_type(totals[f], values[f]) for f in feature_names
    ]
    print('Found {} features: {} categoricals + {} ordinals'.format(
        len(feature_names),
        sum(t is CATEGORICAL for t in feature_types),
        sum(t is ORDINAL for t in feature_types)))

    # Write result.
    with csv_writer(schema_csv_out) as writer:
        writer.writerow(['name', 'type', 'count', 'unique', 'values'])
        for name, typ in zip(feature_names, feature_types):
            row = [name, typ, totals[name], len(values[name])]
            row += values[name]
            writer.writerow(row)


def load_schema(schema_csv_in):
    print('Loading schema from {}'.format(schema_csv_in))

    # Load names, types, and values.
    feature_names = []
    feature_types = {}
    categorical_values = {}
    categorical_index = {}
    ordinal_ranges = {}
    with csv_reader(schema_csv_in) as reader:
        header = next(reader)
        assert header[0].lower() == 'name'
        assert header[1].lower() == 'type'
        assert header[4].lower() == 'values'
        for row in reader:
            if len(row) < 6:
                continue
            name = intern(row[0])
            typename = intern(row[1])
            if not typename:
                continue
            if typename not in VALID_TYPES:
                raise ValueError('Invalid type: {}\n  expected one of: {}'.
                                 format(typename, ', '.join(VALID_TYPES)))
            feature_names.append(name)
            feature_types[name] = typename
            if typename is CATEGORICAL:
                values = tuple(map(intern, row[4:]))
                categorical_values[name] = values
                categorical_index[name] = {v: i for i, v in enumerate(values)}
            elif typename is ORDINAL:
                values = sorted(map(int, row[4:]))
                ordinal_ranges[name] = (values[0], values[-1])
    print('Found {} features'.format(len(feature_names)))
    if not feature_names:
        raise ValueError('Found no features')

    # Create a ragged index.
    ragged_index = np.zeros(len(feature_names) + 1, dtype=np.int32)
    feature_index = {}
    for pos, name in enumerate(feature_names):
        feature_index[name] = pos
        typename = feature_types[name]
        if typename is CATEGORICAL:
            dim = len(categorical_values[name])
        elif typename is ORDINAL:
            dim = 2
        ragged_index[pos + 1] = ragged_index[pos] + dim

    return {
        'feature_names': feature_names,
        'feature_index': feature_index,
        'feature_types': feature_types,
        'categorical_values': categorical_values,
        'categorical_index': categorical_index,
        'ordinal_ranges': ordinal_ranges,
        'ragged_index': ragged_index,
    }


def load_data(schema, data_csv_in, encoding='utf-8'):
    print('Loading data from {}'.format(data_csv_in))
    feature_index = schema['feature_index']
    feature_types = schema['feature_types']
    categorical_index = schema['categorical_index']
    ordinal_ranges = schema['ordinal_ranges']
    ragged_index = schema['ragged_index']
    prototype_row = np.zeros(ragged_index[-1], np.int8)

    # Load data in binary format.
    rows = []
    cells = 0
    with csv_reader(data_csv_in, encoding) as reader:
        header = list(map(intern, next(reader)))
        metas = [None] * len(header)
        for i, name in enumerate(header):
            if name in feature_types:
                metas[i] = (
                    feature_types[name],
                    feature_index[name],
                    categorical_index.get(name),
                    ordinal_ranges.get(name), )
        for external_row in reader:
            internal_row = prototype_row.copy()
            for value, meta in zip(external_row, metas):
                if meta is None or not value:
                    continue
                typename, pos, index, min_max = meta
                if typename is CATEGORICAL:
                    try:
                        value = index[value]
                    except KeyError:
                        continue
                    internal_row[pos + value] = 1
                    cells += 1
                elif typename is ORDINAL:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                    if value < min_max[0] or min_max[1] < value:
                        continue
                    internal_row[pos + 0] = value - min_max[0]
                    internal_row[pos + 1] = min_max[1] - value
                    cells += 1
            rows.append(internal_row)
    print('Loaded {} cells in {} rows, {:0.1f}% observed'.format(
        cells, len(rows), 100.0 * cells / len(rows) / len(feature_types)))
    return np.stack(rows)


@parsable
def import_data(data_csv_in, schema_csv_in, dataset_out, encoding='utf-8'):
    """Import a data.csv file into internal treecat format.

    Common encodings include: utf-8, cp1252.
    """
    schema = load_schema(schema_csv_in)
    data = load_data(schema, data_csv_in, encoding)
    print('Imported {} rows x {} colums'.format(data.shape[0], data.shape[1]))
    dataset = {'schema': schema, 'data': data}
    pickle_dump(dataset, dataset_out)


@parsable
def cat(*paths):
    """Print .pkz files in human readable form."""
    for path in paths:
        assert path.endswith('.pkz')
        print(pickle_load(path))


if __name__ == '__main__':
    parsable()
