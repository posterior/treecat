from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from treecat.format import guess_schema
from treecat.format import load_data
from treecat.format import load_schema
from treecat.testutil import tempdir

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
DATA_CSV = os.path.join(TESTDATA, 'tiny_data.csv')
TYPES_CSV = os.path.join(TESTDATA, 'tiny_types.csv')
VALUES_CSV = os.path.join(TESTDATA, 'tiny_values.csv')


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
