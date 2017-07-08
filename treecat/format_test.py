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
SCHEMA_CSV = os.path.join(TESTDATA, 'tiny_schema.csv')


def test_guess_schema():
    with tempdir() as dirname:
        schema_csv_out = os.path.join(dirname, 'schema.csv')
        guess_schema(DATA_CSV, schema_csv_out)
        actual = open(schema_csv_out).read()
        expected = open(SCHEMA_CSV).read()
        assert actual == expected


def test_load_schema():
    load_schema(SCHEMA_CSV)


def test_load_data():
    schema = load_schema(SCHEMA_CSV)
    load_data(schema, DATA_CSV)
