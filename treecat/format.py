from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
from contextlib import contextmanager

from parsable import parsable

from six.moves import cPickle as pickle

parsable = parsable.Parsable()


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
    raise NotImplementedError()


@parsable
def export_data(dataset_in, schema_csv_out, data_csv_out):
    """Export a treecat dataset file to a schema and data csv."""
    raise NotImplementedError()


if __name__ == '__main__':
    parsable()
