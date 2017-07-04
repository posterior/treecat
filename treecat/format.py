from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import logging
from contextlib import contextmanager

import numpy as np
from parsable import parsable

from six.moves import cPickle as pickle
from six.moves import intern
from six.moves import zip

logger = logging.getLogger(__name__)
parsable = parsable.Parsable()

VALID_TYPES = ('categorical', 'ordinal')


def pickle_dump(data, filename):
    """Pickle data to file using gzip compression."""
    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(filename):
    """Unpickle data from file using gzip compression."""
    with gzip.GzipFile(filename, 'rb') as f:
        return pickle.load(f)


@contextmanager
def csv_reader(filename):
    with open(filename, 'rb') as f:
        yield csv.reader(f)


@contextmanager
def csv_writer(filename):
    with open(filename, 'wb') as f:
        yield csv.writer(f)


@parsable
def import_data(schema_csv_in, data_csv_in, dataset_out):
    """Import a csv file into internal treecat format."""
    # Load schema.
    features = []
    types = {}
    with csv_reader(schema_csv_in) as reader:
        header = reader.next()
        assert header[0].lower() == 'name'
        assert header[1].lower() == 'type'
        for row in reader:
            if len(row) < 2:
                continue
            name = intern(row[0])
            typename = intern(row[1].lower())
            features.append(name)
            if typename not in VALID_TYPES:
                raise ValueError('Invalid type: {}\n  expected one of: {}'.
                                 format(typename, ', '.join(VALID_TYPES)))
            types[name] = typename
    logger.info('Found %d features', len(features))
    if not features:
        raise ValueError('Found no features')

    # Load data.
    rows = []
    all_values = {key: set() for key in types.keys()}
    num_cells = 0
    with csv_reader(data_csv_in) as reader:
        header = list(map(intern, reader.next()))
        row_dict = {}
        for i, row in enumerate(reader):
            for name, value in zip(header, row):
                if value and name in types:
                    value = intern(value)
                    row_dict[name] = value
                    all_values[name].add(value)
            rows.append(row_dict)
            num_cells += len(row_dict)
    logger.info('Found %d rows and %d cells', len(rows), num_cells)

    # Convert to internal format.
    feature_pos = {n: i for i, n in enumerate(features)}
    categorical_values = {
        key: tuple(sorted(values))
        for key, values in all_values.items() if types[key] == 'categorical'
    }
    ordinal_ranges = {
        key: (min(map(int, values)), max(map(int, values)))
        for key, values in all_values.items() if types[key] == 'ordinal'
    }
    ragged_index = np.zeros(len(features) + 1, dtype=np.int32)
    for pos, name in enumerate(features):
        typename = types[name]
        if typename == 'categorical':
            dim = len(all_values[name])
        elif typename == 'ordinal':
            dim = 2
        ragged_index[pos + 1] = ragged_index[pos] + dim
    data = np.zeros([len(rows), ragged_index[-1]], dtype=np.int8)
    for row_id, row in enumerate(rows):
        for name, value in row.items():
            pos = ragged_index[feature_pos[name]]
            typename = types[name]
            if typename == 'categorical':
                data[row_id, pos + categorical_values[name].index(value)] = 1
            elif typename == 'ordinal':
                value = int(value)
                min_value, max_value = ordinal_ranges[name]
                data[row_id, pos] = value - min_value
                data[row_id, pos + 1] = max_value - value
    dataset = {
        'schema': {
            'features': features,
            'types': types,
            'categorical_values': categorical_values,
            'ordinal_ranges': ordinal_ranges,
        },
        'ragged_index': ragged_index,
        'data': data,
    }
    pickle_dump(dataset, dataset_out)


@parsable
def export_data(dataset_in, schema_csv_out, data_csv_out):
    """Export a treecat dataset file to a schema and data csv."""
    dataset = pickle_load(dataset_in)
    schema = dataset['schema']
    features = schema['features']
    types = schema['types']
    categorical_values = schema['categorical_values']
    ordinal_ranges = schema['ordinal_ranges']
    ragged_index = dataset['ragged_index']
    data = dataset['data']

    # Write schema csv.
    with csv_writer(schema_csv_out) as writer:
        writer.write(['name', 'type'])
        for name in features:
            writer.write([name, types[name]])

    # Write data csv.
    N = data.shape[0]
    V = len(features)
    with csv_writer(data_csv_out) as writer:
        writer.writerow(features)
        for row_id in range(N):
            row = [''] * V
            for v in range(V):
                cell = data[row_id, ragged_index[v]:ragged_index[v + 1]]
                if np.all(cell == 0):
                    continue
                typename = types[features[v]]
                if typename == 'categorical':
                    row[v] = categorical_values[name][cell.argmax()]
                elif typename == 'ordinal':
                    min_value, max_value = ordinal_ranges[name]
                    row[v] = str(cell[0] + min_value)
            writer.writerow(row)


@parsable
def cat(*paths):
    """Print .pkl.gz files in human readable form."""
    for path in paths:
        assert path.endswith('.pkl.gz')
        print(pickle_load(path))


if __name__ == '__main__':
    parsable()
