from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest

from treecat.format import guess_schema
from treecat.testutil import tempdir

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')


@pytest.mark.xfail
def test_guess_schema():
    data_csv_in = os.path.join(DATA, 'data.csv')
    with tempdir() as dirname:
        schema_csv_out = os.path.join(dirname, 'schema.csv')
        guess_schema(data_csv_in, schema_csv_out)
        actual = open(os.path.join(dirname, 'schema.csv')).read()
        expected = open(schema_csv_out).read()
        assert actual == expected
