from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from treecat.util import jit


@jit
def jit_add_categorical_data(ragged_index, data, n, v, value):
    """Update multinomial sufficient statistics."""
    data[n, ragged_index[v] + value] += 1


@jit
def jit_add_real_data(data, n, v, value):
    """Update real sufficient statistics."""
    # Adapted from https://www.johndcook.com/blog/standard_deviation
    ss = data[n, v, 1:4]
    if ss[0] == 0:
        ss[0] = 1.0
        ss[1] = value
        ss[2] = 0.0
    else:
        ss[0] += 1.0
        old_diff = value - ss[1]
        ss[1] += old_diff / ss[0]
        new_diff = value - ss[1]
        ss[2] += old_diff * new_diff


class Table(object):
    """Table of row-oriented heterogeneous repeated data."""

    def __init__(self, schema, rows):
        self._schema = dict(schema)
        N = len(rows)

        # Collect categorical features and values.
        categorical_features = sorted(k for k, v in schema.items()
                                      if v == 'categorical')
        categorical_values = [set() for _ in categorical_features]
        for row in rows:
            for name, values in zip(categorical_features, categorical_values):
                try:
                    # TODO Handle repeated values and ordinals.
                    value = row[name]
                except KeyError:
                    continue
                values.add(value)
        categorical_values = [sorted(v) for v in categorical_values]
        categorical_index = [{v: i
                              for i, v in enumerate(vs)}
                             for vs in categorical_values]
        self._categorical_features = categorical_features
        self._categorical_values = categorical_values
        self._categorical_index = categorical_index

        # Create a ragged array of categorical sufficient statistics.
        ragged_index = np.zeros(len(categorical_features) + 1, dtype=np.int32)
        for v, name in enumerate(categorical_features):
            dim = len(categorical_values[v])
            assert dim < 128
            ragged_index[1 + v] = ragged_index[v] + dim
        data = np.zeros((N, ragged_index[-1]), dtype=np.int8)
        for n, row in enumerate(rows):
            for v, name in enumerate(categorical_features):
                try:
                    value = row[name]
                except KeyError:
                    continue
                int_value = categorical_index[v][value]
                jit_add_categorical_data(ragged_index, data, n, v, int_value)
        self._categorical_ragged_index = ragged_index
        self._categorical_data = data

        # Collect real-valued features and create data.
        real_features = sorted(k for k, v in schema.items() if v == 'real')
        data = np.zeros((N, len(real_features), 3), dtype=np.float32)
        for n, row in enumerate(rows):
            for v, name in enumerate(real_features):
                try:
                    value = row[name]
                except KeyError:
                    continue
                jit_add_real_data(data, n, v, value)
        self._real_data = data

    @property
    def categorical_ragged_index(self):
        return self._categorical_ragged_index

    @property
    def categorical_data(self):
        return self._categorical_data

    @property
    def real_data(self):
        return self._real_data
