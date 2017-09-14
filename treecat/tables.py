"""Internal representations of tables of heterogeneous data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from treecat.util import TODO

TY_MULTINOMIAL = 0
TY_CATEGORICAL = 1
TY_ORDINAL = 2


class Table(object):
    """Internal representation of read-only row-oriented heterogeneous data."""

    def __init__(
            self,
            feature_types,
            ragged_index,
            data,
            categorical_data=None, ):
        """Create a read-only table of heterogeneous data.

        Args:
            feature_types: A [V]-shaped numpy array of datatype ids.
            ragged_index: A [V+1]-shaped numpy array of indices into the ragged
                data array.
            data: An [N, _]-shaped numpy array of ragged data, where the vth
                column is stored in data[:, ragged_index[v]:ragged_index[v+1]].
            categorical_data: An [N, Vc]-shaped numpy array of category
                indices, where Vc is the number of categorical features and N
                is the number fo rows.
        """
        N = data.shape[0]
        V = len(feature_types)

        feature_types = np.asarray(feature_types, dtype=np.int8)
        feature_types.flags.writeable = False
        assert len(feature_types.shape) == 1

        # Multinomial features.
        ragged_index = np.asarray(ragged_index, np.int32)
        ragged_index.flags.writeable = False
        assert len(ragged_index.shape) == 1
        assert ragged_index.shape[0] == 1 + V
        data = np.asarray(data, np.int8)
        data.flags.writeable = False
        assert len(data.shape) == 2
        assert data.shape[0] == N
        assert data.shape[1] == ragged_index[-1]

        # Categorical features.
        if categorical_data is None:
            categorical_data = np.empty([N, 0], dtype=np.int16)
        categorical_data = np.asarray(categorical_data, dtype=np.int16)
        assert len(categorical_data.shape) == 2
        assert categorical_data.shape[0] == N

        self._feature_types = feature_types
        self._multinomial_ragged_index = ragged_index
        self._multinomial_data = data
        self._categorical_data = categorical_data

    @property
    def num_rows(self):
        return self._multinomial_data.shape[0]

    @property
    def num_cols(self):
        return self._feature_types.shape[0]

    @property
    def feature_types(self):
        return self._feature_types

    @property
    def ragged_index(self):
        return self._multinomial_ragged_index

    @property
    def data(self):
        return self._multinomial_data

    @property
    def categorical_data(self):
        return self._categorical_data

    @staticmethod
    def concatenate(tables):
        """Concatenate a list of Table objects along rows.

        Args:
            tables: An iterable of Table objects with the same feature types.

        Returns:
            A Table object with the same feature types and with rows from all
            input tables.
        """
        tables = list(tables)
        assert tables
        if len(tables) == 1:
            return tables[0]
        feature_types = table[0].feature_types
        for table in tables[1:]:
            assert np.all(table.feature_types == feature_types)
        multinomial_ragged_index = tables[0].ragged_index
        multinomial_data = np.concatenate([t.data for t in tables])
        categorical_data = np.concatenate([t.categorical_data for t in tables])
        return Table(
            feature_types,
            multinomial_ragged_index,
            multinomial_data,
            categorical_data, )

    def __getitem__(self, index):
        TODO('Handle both single rows and slices')

    def __eq__(self, other):
        assert np.all(self.feature_types == other.feature_types)
        fields = [
            '_multinomial_ragged_index',
            '_multinomial_data',
            '_categorical_data',
        ]
        return all(
            np.all(getattr(self, field) == getattr(other, field))
            for field in fields)


_INTERNALIZE = {
    TY_MULTINOMIAL: TY_MULTINOMIAL,
    TY_CATEGORICAL: TY_MULTINOMIAL,  # TODO Change to TY_CATEGORICAL.
    TY_ORDINAL: TY_MULTINOMIAL,
}


def internalize_table(table):
    """Transform features of a table to types with internal support.

    Args:
        table: A Table object with arbitrary feature types.

    Returns:
        A Table object with transformed feature types.
    """
    feature_types = np.array([_INTERNALIZE[ty]
                              for ty in table.feature_types], np.int8)
    if np.all(feature_types == table.feature_types):
        return table

    for v, (old, new) in enumerate(zip(table.feature_types, feature_types)):
        TODO('convert types')
