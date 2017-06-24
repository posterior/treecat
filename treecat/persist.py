from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
from six.moves import cPickle as pickle


def pickle_dump(data, filename):
    """Pickle data to file using gzip compression."""
    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(filename):
    """Unpickle data from file using gzip compression."""
    with gzip.GzipFile(filename, 'rb') as f:
        return pickle.load(f)
