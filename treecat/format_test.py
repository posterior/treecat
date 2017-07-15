from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
import pytest

from treecat.format import export_rows
from treecat.format import guess_schema
from treecat.format import import_rows
from treecat.format import load_data
from treecat.format import load_schema
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.testutil import TESTDATA
from treecat.testutil import assert_equal
from treecat.testutil import tempdir

DATA_CSV = os.path.join(TESTDATA, 'tiny_data.csv')
TYPES_CSV = os.path.join(TESTDATA, 'tiny_types.csv')
VALUES_CSV = os.path.join(TESTDATA, 'tiny_values.csv')

EXAMPLE_DATA = [
    u'foo',
    [1, 2, 3],
    {
        u'foo': 0
    },
    np.array([[0, 1], [2, 3]], dtype=np.int8),
]


@pytest.mark.parametrize('data,ext',
                         itertools.product(EXAMPLE_DATA, ['pkz', 'jz']))
def test_pickle(data, ext):
    with tempdir() as dirname:
        filename = os.path.join(dirname, 'test.{}'.format(ext))
        pickle_dump(data, filename)
        actual = pickle_load(filename)
        assert_equal(actual, data)


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
    load_schema(TYPES_CSV, VALUES_CSV)


def test_load_data():
    schema = load_schema(TYPES_CSV, VALUES_CSV)
    load_data(schema, DATA_CSV)


def test_export_import_rows():
    schema = load_schema(TYPES_CSV, VALUES_CSV)
    data = load_data(schema, DATA_CSV)
    print(schema['feature_index'])
    print(data)
    rows = export_rows(schema, data)
    assert len(rows) == data.shape[0]
    actual_data = import_rows(schema, rows)
    print(actual_data)
    assert np.all(actual_data == data)
