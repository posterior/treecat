from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from treecat.util import jit


class Schema(object):
    """Schema describing data types of multivariate heterogeneous data."""

    dtypes = ('multinomial', 'normal')
    dtype_index = {dtypes.index(dtype) for dtype in dtypes}

    def __init__(self):
        self._multinomial_features = []  # list(string).
        self._multinomial_values = {}  # dict(string, list(string)).
        self._multinomial_value_index = {}  # dict(string, dict(string, int)).
        self._normal_features = []  # list(string).

    def add_multinomial_feature(self, name):
        assert name not in self._multinomial_features
        self._multinomial_features.append(name)

    def add_multinomial_value(self, name, value):
        values = self._multinomial_values[name]
        value_index = self._multinomial_value_index[name]
        if value not in value_index:
            value_index[value] = len(values)
            values.append(value)

    def add_normal_feature(self, name):
        assert name not in self._normal_features
        self._normal_features.append(name)


class Table(object):
    """Table of row-oriented heterogeneous repeated data."""

    def __init__(self, schema, rows):
        self._schema = dict(schema)  # TODO Use a Schema object.
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
    def schema(self):
        return self._schema

    @property
    def categorical_ragged_index(self):
        return self._categorical_ragged_index

    @property
    def categorical_data(self):
        return self._categorical_data

    @property
    def real_data(self):
        return self._real_data

    def __add__(self, other):
        """Combine observations of two row-aligned datasets.

        This adds data in the sense of repeated observation of each row.

        See also `Table.cat()` for combining different rows
        and `Table.join` for combining different columns.

        Args:
            other (Table): Another table with the same schema and aligned rows.

        Returns:
            Table: A table with the same schema and same number of rows.
        """
        raise NotImplementedError


@jit
def jit_add_categorical_data(ragged_index, data, n, v, value):
    """Update multinomial sufficient statistics."""
    data[n, ragged_index[v] + value] += 1


@jit
def jit_add_real_data(data, n, v, value):
    """Update real sufficient statistics."""
    # Adapted from https://www.johndcook.com/blog/standard_deviation
    ss = data[n, v, 0:3]
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
