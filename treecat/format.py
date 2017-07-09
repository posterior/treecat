from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import io
import re
import sys
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from parsable import parsable

from six.moves import cPickle as pickle
from six.moves import intern
from six.moves import zip

parsable = parsable.Parsable()

CATEGORICAL = intern('categorical')
ORDINAL = intern('ordinal')
VALID_TYPES = (CATEGORICAL, ORDINAL)
MAX_CATEGORIES = 20
NA_STRINGS = frozenset([intern(''), intern('null'), intern('none')])
OTHER = intern('_OTHER')


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
        if sys.version_info >= (3, 0):
            yield csv.reader(f)
        else:
            yield csv.reader(line.encode(encoding, 'ignore') for line in f)


@contextmanager
def csv_writer(filename):
    with open(filename, 'w') as f:
        yield csv.writer(f)


def normalize_string(string):
    return re.sub(r'\s+', ' ', string.strip().lower())


def is_small_int(value):
    try:
        int_value = int(value)
        return 0 <= int_value and int_value <= MAX_CATEGORIES
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
    if len(values) <= MAX_CATEGORIES:
        if all(is_small_int(v) for (v, c) in values if v is not OTHER):
            return ORDINAL
    if len(values) <= min(count / 2, MAX_CATEGORIES):
        return CATEGORICAL
    return ''


@parsable
def guess_schema(data_csv_in, types_csv_out, values_csv_out, encoding='utf-8'):
    """Create a best-guess types and values for a given dataset.

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
    uniques = defaultdict(lambda: 0)
    for name, counts in values.items():
        uniques[name] = len(counts)

    # Exclude singleton values, because they provide no statistical value
    # and they often leak identifying info.
    singletons = Counter()
    for name in feature_names:
        counts = values[name]
        singles = [v for v, c in counts.items() if c == 1]
        for value in singles:
            del counts[value]
            counts[OTHER] += 1
            singletons[name] += 1
        values[name] = counts.most_common(MAX_CATEGORIES)
        values[name].sort(key=lambda vc: (-vc[1], vc[0]))  # Brake ties.

    # Guess feature types.
    feature_types = [
        guess_feature_type(totals[f], values[f]) for f in feature_names
    ]
    print('Found {} features: {} categoricals + {} ordinals'.format(
        len(feature_names),
        sum(t is CATEGORICAL for t in feature_types),
        sum(t is ORDINAL for t in feature_types)))

    # Write types.
    with csv_writer(types_csv_out) as writer:
        writer.writerow(['name', 'type', 'total', 'unique', 'singletons'])
        for name, typ in zip(feature_names, feature_types):
            writer.writerow(
                [name, typ, totals[name], uniques[name], singletons[name]])

    # Write values.
    with csv_writer(values_csv_out) as writer:
        writer.writerow(['name', 'value', 'count'])
        for name, typ in zip(feature_names, feature_types):
            for value, count in values[name]:
                writer.writerow([name, str(value), str(count)])


def load_schema(types_csv_in, values_csv_in, encoding='utf-8'):
    print('Loading schema from {} and {}'.format(types_csv_in, values_csv_in))

    # Load types.
    feature_names = []
    feature_types = {}
    with csv_reader(types_csv_in, encoding) as reader:
        header = next(reader)
        assert header[0].lower() == 'name'
        assert header[1].lower() == 'type'
        for row in reader:
            if len(row) < 2:
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

    # Load values.
    categorical_values = defaultdict(list)
    ordinal_values = defaultdict(list)
    with csv_reader(values_csv_in, encoding) as reader:
        header = next(reader)
        assert header[0].lower() == 'name'
        assert header[1].lower() == 'value'
        for row in reader:
            if len(row) < 2:
                continue
            name = intern(row[0])
            if name not in feature_types:
                continue
            value = intern(row[1])
            typename = feature_types[name]
            if typename is CATEGORICAL:
                categorical_values[name].append(value)
            elif typename is ORDINAL:
                if value is not OTHER:
                    ordinal_values[name].append(int(value))
            else:
                raise ValueError(typename)
    print('Found {} features'.format(len(feature_names)))
    if not feature_names:
        raise ValueError('Found no features')

    # Create value indices.
    categorical_index = {}
    ordinal_ranges = {}
    for name, typename in feature_types.items():
        if typename is CATEGORICAL:
            values = tuple(categorical_values[name])
            categorical_values[name] = values
            categorical_index[name] = {v: i for i, v in enumerate(values)}
        elif typename is ORDINAL:
            values = sorted(ordinal_values[name])
            ordinal_ranges[name] = (values[0], values[-1])
        else:
            raise ValueError(typename)

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
                metas[i] = (feature_types[name], feature_index[name],
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
def import_data(data_csv_in,
                types_csv_in,
                values_csv_in,
                dataset_out,
                encoding='utf-8'):
    """Import a data.csv file into internal treecat format.

    Common encodings include: utf-8, cp1252.
    """
    schema = load_schema(types_csv_in, values_csv_in, encoding)
    data = load_data(schema, data_csv_in, encoding)
    print('Imported data shape: [{}, {}]'.format(data.shape[0], data.shape[1]))
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
