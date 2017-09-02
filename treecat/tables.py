"""Internal representations of heterogeous data and statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import gammaln

from treecat.util import jit


class Schema(object):
    """Schema describing a data types of multivariate heterogeneous data."""

    DTYPES = ('categorical', 'normal')
    DTYPE_INDEX = {DTYPES.index(dtype) for dtype in DTYPES}

    def __init__(self, feature_types):
        assert isinstance(feature_types, dict)

        self._feature_types = dict(feature_types)
        self._normal_features = tuple(
            sorted([
                name for name, dtype in feature_types.items
                if dtype == 'normal'
            ]))
        self._normal_index = {
            i: name
            for name, i in enumerate(self._normal_features)
        }
        self._cat_values = {}  # dict(string, list(string)).
        self._cat_value_index = {}  # dict(string, dict(string, int)).

    def register_data(self, rows):
        for row in rows:
            for name, value in row.items():
                if self._feature_types[name] == 'categorical':
                    self.register_cat_value(name, value)

    def register_cat_value(self, name, value):
        values = self._cat_values[name]
        value_index = self._cat_value_index[name]
        if value not in value_index:
            value_index[value] = len(values)
            values.append(value)


class ValueTable(object):
    """Table of row-oriented heterogeneous single-observation data."""

    def __init__(self, schema, num_rows):
        assert isinstance(schema, Schema)
        self._schema = schema
        N = num_rows
        Vr = len(schema._real_features)
        Vc = len(schema._cat_features)

        self._cat_data = np.zeros([N, Vc], dtype=np.int8)
        self._real_data = np.zeros([N, Vr], dtype=np.float32)


def gaussian_logprob_data(count, mean, count_times_variance, prior):
    """
    References:
    - Kevin Murphy (2007)
      "Conjugate Bayesian analysis of the Gaussian distribution"
      http://www-devel.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """
    raise NotImplementedError


class JointStats(object):
    """Sufficient statistics for a joint distribution of categoricals + reals.

    This represents sufficient statistics for a mixture-of-Gaussians model of
    heterogeneous categorical and real data, where the categoricals comprise a
    mixture id and the reals are jointly modeled as a MVN conditional on the
    categorical data. This assumes that all data is observed simultaneously.

    The sufficient statistics are as follows:
    - _ss_0 = number of observations of each mixture component.
    - _ss_1 = means of each mixture component.
    - _ss_2 = count * variance of each mixture component.
    """

    def __init__(self, cat_dims, real_dim):
        cat_dim = np.product(cat_dims, np.int32)
        self._cat_dims = cat_dims
        self._flatten = np.cumprod(cat_dims) / cat_dims
        self._ss_0 = np.zeros([cat_dim], np.int32)
        self._ss_1 = np.zeros([cat_dim, real_dim], np.float32)
        self._ss_2 = np.zeros([cat_dim, real_dim, real_dim], np.float32)

    def reset(self):
        self._ss_0[...] = 0
        self._ss_1[...] = 0
        self._ss_2[...] = 0

    def add_row(self, cat_row, real_row):
        assert isinstance(cat_row, np.ndarray)
        assert isinstance(real_row, np.ndarray)
        assert cat_row.shape == (len(self.cat_dims), )
        assert real_row.shape == (self.real_dim, )
        m = np.dot(self._flatten, cat_row)
        self._ss_0[m] += 1
        n = np.float32(self._ss_0[m])
        one_over_n = 1.0 / n
        diff = real_row - self._ss_1[m]
        self._ss_1[m] += diff * one_over_n
        self._ss_2 += np.outer(diff, diff) * ((1.0 + n) * one_over_n)

    def remove_row(self, cat_row, real_row):
        assert isinstance(cat_row, np.ndarray)
        assert isinstance(real_row, np.ndarray)
        raise NotImplementedError

    def set_from_table(self, cat_table, real_table):
        assert isinstance(cat_table, np.ndarray)
        assert isinstance(real_table, np.ndarray)
        assert cat_table.shape[0] == real_table.shape[0]
        self.reset()
        for n in range(cat_table.shape[0]):
            self.add_row(cat_table[n, :], real_table[n, :])

    def logprob(self, cat_prior, real_prior):
        result = 0.0
        for m in range(self._ss_0.shape[0]):
            result += gammaln(self._ss_0[m] + cat_prior)
            result += gaussian_logprob_data(self._ss_0[m], self._ss_1[m],
                                            self._ss_2[m], real_prior)
        return result


class EdgesStats(object):
    """Sufficient statistics for a joint distribution of a pair of variables.

    This should be distributionally eqivalent to JointStats, but with variables
    separated into two sets, say head and tail.
    """

    def __init__(self, head_cat_dim, head_real_dim, tail_cat_dim,
                 tail_real_dim):
        assert head_cat_dim >= 1
        assert tail_cat_dim >= 1
        raise NotImplementedError

    def filter_forward(self, head_message):
        raise NotImplementedError

    def filter_backward(self, head_message):
        raise NotImplementedError

    def sample_forward(self, tail_sample):
        raise NotImplementedError

    def sample_backward(self, tail_sample):
        raise NotImplementedError

    def logprob(self, head_cat_prior, head_real_prior, tail_cat_prior,
                tail_real_prior):
        raise NotImplementedError


class StatsTable(object):
    """Table of row-oriented heterogeneous missing/repeated data."""

    def __init__(self, schema, rows):
        assert isinstance(schema, Schema)
        self._schema = schema
        N = len(rows)

        # Collect categorical features and values.
        cat_features = sorted(k for k, v in schema.items()
                              if v == 'categorical')
        cat_values = [set() for _ in cat_features]
        for row in rows:
            for name, values in zip(cat_features, cat_values):
                try:
                    # TODO Handle repeated values and ordinals.
                    value = row[name]
                except KeyError:
                    continue
                values.add(value)
        cat_values = [sorted(v) for v in cat_values]
        cat_index = [{v: i for i, v in enumerate(vs)} for vs in cat_values]
        self._cat_features = cat_features
        self._cat_values = cat_values
        self._cat_index = cat_index

        # Create a ragged array of categorical sufficient statistics.
        ragged_index = np.zeros(len(cat_features) + 1, dtype=np.int32)
        for v, name in enumerate(cat_features):
            dim = len(cat_values[v])
            assert dim < 128
            ragged_index[1 + v] = ragged_index[v] + dim
        data = np.zeros((N, ragged_index[-1]), dtype=np.int8)
        for n, row in enumerate(rows):
            for v, name in enumerate(cat_features):
                try:
                    value = row[name]
                except KeyError:
                    continue
                int_value = cat_index[v][value]
                jit_add_cat_data(ragged_index, data, n, v, int_value)
        self._cat_ragged_index = ragged_index
        self._cat_data = data

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
    def num_rows(self):
        return self._real_data.shape[0]

    @property
    def cat_ragged_index(self):
        return self._cat_ragged_index

    @property
    def cat_data(self):
        return self._cat_data

    @property
    def real_data(self):
        return self._real_data

    def __add__(self, other):
        """Combine observations of two row-aligned datasets.

        This adds data in the sense of repeated observation of each row.

        See also `StatsTable.cat()` for combining different rows
        and `StatsTable.join` for combining different columns.

        Args:
            other (StatsTable): Another StatsTable with the same schema and
                aligned rows.

        Returns:
            StatsTable: A table with the same schema and same number of rows.
        """
        assert isinstance(other, StatsTable)
        assert other.num_rows == self.num_rows
        raise NotImplementedError


@jit
def jit_add_cat_data(ragged_index, data, n, v, value):
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
