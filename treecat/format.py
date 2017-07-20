from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import functools
import gzip
import hashlib
import io
import logging
import os
import re
import shutil
import sys
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager

import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
from parsable import parsable

from six.moves import cPickle as pickle
from six.moves import range
from six.moves import zip

logger = logging.getLogger(__name__)
parsable = parsable.Parsable()
jsonpickle.ext.numpy.register_handlers()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMO_STORE = os.path.join(REPO, 'data', 'memoize')

CATEGORICAL = 'categorical'
ORDINAL = 'ordinal'
VALID_TYPES = (CATEGORICAL, ORDINAL)
MAX_CATEGORIES = 20
NA_STRINGS = {
    'null': '',
    'none': '',
}


def json_dumps(value):
    return jsonpickle.encode(value)


def json_loads(string):
    return jsonpickle.decode(string)


def pickle_dump(data, filename):
    """Serialize data to file using gzip compression."""
    if filename.endswith('.pkz'):
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=2)  # Try to support python 2.
    elif filename.endswith('.jz'):
        with gzip.open(filename, 'wt') as f:
            f.write(json_dumps(data))
    else:
        raise ValueError(
            'Cannot determine format: {}'.format(os.path.basename(filename)))


def pickle_load(filename):
    """Deserialize data from file using gzip compression."""
    if filename.endswith('.pkz'):
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.jz'):
        with gzip.open(filename, 'rt') as f:
            return json_loads(f.read())
    else:
        raise ValueError(
            'Cannot determine format: {}'.format(os.path.basename(filename)))


def fingerprint(obj):
    serialized = json_dumps(obj)
    hasher = hashlib.sha1()
    try:
        hasher.update(serialized)
    except TypeError:
        hasher.update(serialized.encode('utf-8'))
    return hasher.hexdigest()


def pickle_memoize(fun):
    if not os.path.exists(MEMO_STORE):
        os.makedirs(MEMO_STORE)
    template = os.path.join(MEMO_STORE, '{}.{}.{{}}.pkz'.format(
        fun.__module__, fun.__name__))

    @functools.wraps(fun)
    def decorated(*args):
        memo_path = template.format(fingerprint(args))
        if os.path.exists(memo_path):
            return pickle_load(memo_path)
        else:
            value = fun(*args)
            pickle_dump(value, memo_path)
            return value

    return decorated


@parsable
def clean():
    """Clean pickle_memoized cache."""
    if os.path.exists(MEMO_STORE):
        shutil.rmtree(MEMO_STORE)


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
    string = re.sub(r'\s+', ' ', string.strip().lower())
    return NA_STRINGS.get(string, string)


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
        if all(is_small_int(v) for (v, c) in values):
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
        feature_names = list(next(reader))
        for row in reader:
            for name, value in zip(feature_names, row):
                value = normalize_string(value)
                if not value:
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
            singletons[name] += 1
        values[name] = counts.most_common(MAX_CATEGORIES)
        values[name].sort(key=lambda vc: (-vc[1], vc[0]))  # Brake ties.

    # Guess feature types.
    feature_types = [
        guess_feature_type(totals[f], values[f]) for f in feature_names
    ]
    print('Found {} features: {} categoricals + {} ordinals'.format(
        len(feature_names),
        sum(t == CATEGORICAL for t in feature_types),
        sum(t == ORDINAL for t in feature_types)))

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
            name = row[0]
            typename = row[1]
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
            name = row[0]
            if name not in feature_types:
                continue
            value = row[1]
            typename = feature_types[name]
            if typename == CATEGORICAL:
                categorical_values[name].append(value)
            elif typename == ORDINAL:
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
        if typename == CATEGORICAL:
            values = tuple(categorical_values[name])
            categorical_values[name] = values
            categorical_index[name] = {v: i for i, v in enumerate(values)}
        elif typename == ORDINAL:
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
        if typename == CATEGORICAL:
            dim = len(categorical_values[name])
        elif typename == ORDINAL:
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
    column_counts = Counter()
    with csv_reader(data_csv_in, encoding) as reader:
        header = list(next(reader))
        metas = [None] * len(header)
        for i, name in enumerate(header):
            if name in feature_types:
                metas[i] = (  #
                    name, feature_types[name],
                    ragged_index[feature_index[name]],
                    categorical_index.get(name), ordinal_ranges.get(name), )
        for external_row in reader:
            internal_row = prototype_row.copy()
            for value, meta in zip(external_row, metas):
                if meta is None:
                    continue
                value = normalize_string(value)
                if not value:
                    continue
                name, typename, pos, index, min_max = meta
                if typename == CATEGORICAL:
                    try:
                        value = index[value]
                    except KeyError:
                        continue
                    internal_row[pos + value] = 1
                elif typename == ORDINAL:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                    if value < min_max[0] or min_max[1] < value:
                        continue
                    internal_row[pos + 0] = value - min_max[0]
                    internal_row[pos + 1] = min_max[1] - value
                else:
                    raise ValueError(typename)
                cells += 1
                column_counts[name] += 1
            rows.append(internal_row)
    print('Loaded {} cells in {} rows, {:0.1f}% observed'.format(
        cells, len(rows), 100.0 * cells / len(rows) / len(feature_types)))
    for name in feature_types.keys():
        if column_counts[name] == 0:
            print('WARNING: No values found for feature {}'.format(name))
    return np.stack(rows)


def import_rows(schema, rows):
    """Import multiple rows of json data to internal format.

    Args:
      schema: A schema dict as returned by load_schema().
      rows: A N-long list of sparse dicts mapping feature names to values,
        where N is the number of rows. Extra keys and invalid values will be
        silently ignored.

    Returns:
      An [N, R]-shaped numpy array of ragged data, where N is the number of
      rows and R = schema['ragged_index'][-1].
    """
    logger.debug('Importing {:d} rows', len(rows))
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)
    feature_index = schema['feature_index']
    feature_types = schema['feature_types']
    categorical_index = schema['categorical_index']
    ordinal_ranges = schema['ordinal_ranges']
    ragged_index = schema['ragged_index']

    N = len(rows)
    R = ragged_index[-1]
    data = np.zeros([N, R], dtype=np.int8)
    for external_row, internal_row in zip(rows, data):
        for name, value in external_row.items():
            try:
                pos = ragged_index[feature_index[name]]
            except KeyError:
                continue
            typename = feature_types[name]
            if typename == CATEGORICAL:
                index = categorical_index[name]
                try:
                    value = index[value]
                except KeyError:
                    continue
                internal_row[pos + value] = 1
            elif typename == ORDINAL:
                min_max = ordinal_ranges[name]
                try:
                    value = int(value)
                except ValueError:
                    continue
                if value < min_max[0] or min_max[1] < value:
                    continue
                internal_row[pos + 0] = value - min_max[0]
                internal_row[pos + 1] = min_max[1] - value
            else:
                raise ValueError(typename)
    return data


def export_rows(schema, data):
    """Export multiple rows of internal data to json format.

    Args:
      schema: A schema dict as returned by load_schema().
      data: An [N, R]-shaped numpy array of ragged data, where N is the number
        of rows and R = schema['ragged_index'][-1].

    Returns:
      A N-long list of sparse dicts mapping feature names to json values,
      where N is the number of rows.
    """
    logger.debug('Exporting {:d} rows', data.shape[0])
    assert data.dtype == np.int8
    assert len(data.shape) == 2
    ragged_index = schema['ragged_index']
    assert data.shape[1] == ragged_index[-1]
    feature_names = schema['feature_names']
    feature_types = schema['feature_types']
    categorical_values = schema['categorical_values']
    ordinal_ranges = schema['ordinal_ranges']

    rows = [{} for _ in range(data.shape[0])]
    for external_row, internal_row in zip(rows, data):
        for v, name in enumerate(feature_names):
            beg, end = ragged_index[v:v + 2]
            internal_cell = internal_row[beg:end]
            if np.all(internal_cell == 0):
                continue
            typename = feature_types[name]
            if typename == CATEGORICAL:
                assert internal_cell.sum() == 1, internal_cell
                value = categorical_values[name][internal_cell.argmax()]
            elif typename == ORDINAL:
                min_max = ordinal_ranges[name]
                assert internal_cell.sum() == min_max[1] - min_max[0]
                value = internal_cell[0] + min_max[0]
            else:
                raise ValueError(typename)
            external_row[name] = value
    return rows


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
