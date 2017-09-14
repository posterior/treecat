"""Internal representations of tables of heterogeneous data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

TY_MULTINOMIAL = 0


class Table(object):
    """Internal representation of read-only row-oriented heterogeneous data."""

    def __init__(self, feature_types, ragged_index, data):
        """Create a read-only table of heterogeneous data.

        Args:
            feature_types: A [V]-shaped numpy array of datatype ids.
            ragged_index: A [V+1]-shaped numpy array of indices into the ragged
                data array.
            data: An [N, _]-shaped numpy array of ragged data, where the vth
                column is stored in data[:, ragged_index[v]:ragged_index[v+1]].
        """
        feature_types = np.asarray(feature_types, dtype=np.int8)
        feature_types.flags.writeable = False
        assert len(feature_types.shape) == 1

        ragged_index = np.asarray(ragged_index, np.int32)
        ragged_index.flags.writeable = False
        assert len(ragged_index.shape) == 1
        assert ragged_index.shape[0] == 1 + feature_types.shape[0]

        data = np.asarray(data, np.int8)
        data.flags.writeable = False
        assert len(data.shape) == 2
        assert data.shape[1] == ragged_index[-1]

        self._feature_types = feature_types
        self._ragged_index = ragged_index
        self._data = data

    @property
    def feature_types(self):
        return self._feature_types

    @property
    def ragged_index(self):
        return self._ragged_index

    @property
    def data(self):
        return self._data

    @property
    def num_rows(self):
        return self._data.shape[0]

    @property
    def num_cols(self):
        return self._feature_types.shape[0]
