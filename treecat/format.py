from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import hashlib
import io
import logging
import os
import re
import sys
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager

import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import pandas as pd
from parsable import parsable

from six.moves import cPickle as pickle
from six.moves import range
from six.moves import zip
from treecat.structure import find_complete_edge

logger = logging.getLogger(__name__)
parsable = parsable.Parsable()
jsonpickle.ext.numpy.register_handlers()

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
    with open('/tmp/v1.json', 'w') as f:
        print('see /tmp/v1.json')
        f.write(serialized)
    hasher = hashlib.sha1()
    try:
        hasher.update(serialized)
    except TypeError:
        hasher.update(serialized.encode('utf-8'))
    return hasher.hexdigest()


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


def read_csv(filename, encoding='utf-8'):
    return pd.read_csv(filename, dtype=object, encoding=encoding)


def pd_outer_join(dfs, on):
    """Outer-join an iterable of pandas dataframes on a given column.

    Args:
        dfs: A pandas dataframe.
        on: A column name or list of column names.

    Returns:
        A pandas dataframe whose columns are the union of columns in dfs, and
        whose rows are the union of rows joined on 'on'.
    """
    result = dfs[0].set_index(on)
    for i, df in enumerate(dfs[1:]):
        assert not any(col.endswith('_JOIN_') for col in result.columns)
        result = result.join(df.set_index(on), how='outer', rsuffix='_JOIN_')
        for right in result.columns:
            if right.endswith('_JOIN_'):
                left = right[:-6]
                if left in df.columns:
                    result[left].fillna(result[right], inplace=True)
                    del result[right]
                else:
                    result.rename(columns={right: left})
    result = result.sort_index(axis=1)
    return result


@parsable
def join_csvs(column,
              csvs_in,
              csv_out,
              encoding_in='utf-8',
              encoding_out='utf-8'):
    """Outer join a comma-delimited list of csvs on a given column.

    Common encodings include: utf-8, cp1252.
    """
    dfs = [read_csv(csv_in, encoding_in) for csv_in in csvs_in.split(',')]
    df = pd_outer_join(dfs, column)
    df.to_csv(csv_out, encoding=encoding_out)


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
def guess_schema(data_csvs_in, types_csv_out, values_csv_out,
                 encoding='utf-8'):
    """Create a best-guess types and values for a given dataset.

    Common encodings include: utf-8, cp1252.
    """
    print('Guessing schema')

    # Collect statistics.
    totals = Counter()
    values = defaultdict(Counter)
    feature_names = set()
    sources = defaultdict(list)
    for data_csv_in in data_csvs_in.split(','):
        print('reading {}'.format(data_csv_in))
        with csv_reader(data_csv_in, encoding) as reader:
            header = list(next(reader))
            feature_names |= set(header)
            for name in feature_names:
                sources[name].append(os.path.basename(data_csv_in))
            for row in reader:
                for name, value in zip(header, row):
                    value = normalize_string(value)
                    if not value:
                        continue
                    totals[name] += 1
                    values[name][value] += 1
    uniques = defaultdict(lambda: 0)
    for name, counts in values.items():
        uniques[name] = len(counts)
    feature_names = sorted(feature_names)

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
        writer.writerow([
            'name',
            'type',
            'total',
            'unique',
            'singletons',
            'source',
        ])
        for name, typ in zip(feature_names, feature_types):
            writer.writerow([
                name,
                typ,
                totals[name],
                uniques[name],
                singletons[name],
                ','.join(sources[name]),
            ])

    # Write values.
    with csv_writer(values_csv_out) as writer:
        writer.writerow(['name', 'value', 'count'])
        for name, typ in zip(feature_names, feature_types):
            for value, count in values[name]:
                writer.writerow([name, str(value), str(count)])


def load_schema(types_csv_in, values_csv_in, groups_csv_in, encoding='utf-8'):
    print('Loading schema from {}, {}, {}'.format(types_csv_in, values_csv_in,
                                                  groups_csv_in))

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

    # Load optional groups.
    # These are arranged as blocks from sources to targets with constant logit.
    # The logits, sources, and targets can be specified in any order.
    group_logits = defaultdict(lambda: 0)
    group_sources = defaultdict(set)
    group_targets = defaultdict(set)
    if groups_csv_in:
        with csv_reader(groups_csv_in, encoding) as reader:
            header = next(reader)
            assert header[0].lower() == 'group'
            assert header[1].lower() == 'logit'
            assert header[2].lower() == 'source'
            assert header[3].lower() == 'target'
            for row in reader:
                if len(row) < 4:
                    continue
                group, logit, source, target = row[:3]
                if logit:
                    group_logits[group] = float(logit)
                if source:
                    group_sources[group].add(source)
                if target:
                    group_targets[group].add(target)

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

    # Create a tree prior.
    V = len(feature_names)
    K = V * (V - 1) // 2
    tree_prior = np.zeros(K, np.float32)
    for group, logit in group_logits.items():
        for source in group_sources[group]:
            v1 = feature_index[source]
            for target in group_targets[group]:
                v2 = feature_index[target]
                k = find_complete_edge(v1, v2)
                tree_prior[k] = logit

    return {
        'feature_names': feature_names,
        'feature_index': feature_index,
        'feature_types': feature_types,
        'categorical_values': categorical_values,
        'categorical_index': categorical_index,
        'ordinal_ranges': ordinal_ranges,
        'ragged_index': ragged_index,
        'tree_prior': tree_prior,
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
            rows.append(internal_row)
    print('Loaded {} cells in {} rows, {:0.1f}% observed'.format(
        cells, len(rows), 100.0 * cells / len(rows) / len(feature_types)))
    return np.stack(rows)


def import_rows(schema, rows):
    """Import multiple rows of json data to internal format.

    Args:
        schema: A schema dict as returned by load_schema().
        rows: A N-long list of sparse dicts mapping feature names to values,
            where N is the number of rows. Extra keys and invalid values will
            be silently ignored.

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
        data: An [N, R]-shaped numpy array of ragged data, where N is the
            number of rows and R = schema['ragged_index'][-1].

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
def import_data(data_csvs_in,
                types_csv_in,
                values_csv_in,
                groups_csv_in,
                dataset_out,
                encoding='utf-8'):
    """Import a comma-delimited list of csv files into internal treecat format.

    Common encodings include: utf-8, cp1252.
    """
    schema = load_schema(types_csv_in, values_csv_in, groups_csv_in, encoding)
    data = np.concatenate([
        load_data(schema, data_csv_in, groups_csv_in, encoding)
        for data_csv_in in data_csvs_in.split(',')
    ])
    print('Imported data shape: [{}, {}]'.format(data.shape[0], data.shape[1]))
    ragged_index = schema['ragged_index']
    for v, name in enumerate(schema['feature_names']):
        beg, end = ragged_index[v:v + 2]
        count = np.count_nonzero(data[:, beg:end].max(1))
        if count == 0:
            print('WARNING: No values found for feature {}'.format(name))
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
