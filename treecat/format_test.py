from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import pytest

from treecat.format import export_rows
from treecat.format import guess_schema
from treecat.format import import_rows
from treecat.format import load_data
from treecat.format import load_schema
from treecat.format import pd_outer_join
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.testutil import TESTDATA
from treecat.testutil import assert_equal
from treecat.testutil import tempdir

DATA_CSV = os.path.join(TESTDATA, 'tiny_data.csv')
TYPES_CSV = os.path.join(TESTDATA, 'tiny_types.csv')
VALUES_CSV = os.path.join(TESTDATA, 'tiny_values.csv')
GROUPS_CSV = os.path.join(TESTDATA, 'tiny_groups.csv')

EXAMPLE_DATA = [
    u'foo',
    [1, 2, 3],
    {
        u'foo': 0
    },
    np.array([[0, 1], [2, 3]], dtype=np.int8),
]


@pytest.mark.parametrize('data', EXAMPLE_DATA)
@pytest.mark.parametrize('ext', ['pkz', 'jz'])
def test_pickle(data, ext):
    with tempdir() as dirname:
        filename = os.path.join(dirname, 'test.{}'.format(ext))
        pickle_dump(data, filename)
        actual = pickle_load(filename)
        assert_equal(actual, data)


def test_pd_outer_join():
    dfs = [
        pd.DataFrame({
            'id': [0, 1, 2, 3],
            'a': ['foo', 'bar', 'baz', np.nan],
            'b': ['panda', 'zebra', np.nan, np.nan],
        }),
        pd.DataFrame({
            'id': [1, 2, 3, 4],
            'b': ['mouse', np.nan, 'tiger', 'egret'],
            'c': ['toe', 'finger', 'nose', np.nan],
        }),
    ]
    expected = pd.DataFrame({
        'id': [0, 1, 2, 3, 4],
        'a': ['foo', 'bar', 'baz', np.nan, np.nan],
        'b': ['panda', 'zebra', np.nan, 'tiger', 'egret'],
        'c': [np.nan, 'toe', 'finger', 'nose', np.nan],
    }).set_index('id')
    actual = pd_outer_join(dfs, on='id')
    print(expected)
    print(actual)
    assert expected.equals(actual)


def test_guess_schema():
    with tempdir() as dirname:
        types_csv_out = os.path.join(dirname, 'types.csv')
        values_csv_out = os.path.join(dirname, 'values.csv')
        guess_schema(DATA_CSV, types_csv_out, values_csv_out)
        expected_types = open(TYPES_CSV).read()
        expected_values = open(VALUES_CSV).read()
        actual_types = open(types_csv_out).read()
        actual_values = open(values_csv_out).read()
        assert actual_types == expected_types
        assert actual_values == expected_values


def test_load_schema():
    load_schema(TYPES_CSV, VALUES_CSV, GROUPS_CSV)


def test_load_data():
    schema = load_schema(TYPES_CSV, VALUES_CSV, GROUPS_CSV)
    load_data(schema, DATA_CSV)


def test_export_import_rows():
    schema = load_schema(TYPES_CSV, VALUES_CSV, GROUPS_CSV)
    data = load_data(schema, DATA_CSV)
    print(schema['feature_index'])
    print(data)
    rows = export_rows(schema, data)
    assert len(rows) == data.shape[0]
    actual_data = import_rows(schema, rows)
    print(actual_data)
    assert np.all(actual_data == data)
