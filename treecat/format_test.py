from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from treecat.format import guess_schema
from treecat.testutil import tempdir

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')


def test_guess_schema():
    data_csv_in = os.path.join(TESTDATA, 'tiny_data.csv')
    with tempdir() as dirname:
        schema_csv_out = os.path.join(dirname, 'schema.csv')
        guess_schema(data_csv_in, schema_csv_out)
        actual = open(schema_csv_out).read()
        expected = open(os.path.join(TESTDATA, 'tiny_schema.csv')).read()
        assert actual == expected
