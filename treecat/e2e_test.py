from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest

from treecat.format import guess_schema
from treecat.format import load_data
from treecat.format import load_schema
from treecat.serving import serve_model
from treecat.testutil import TESTDATA
from treecat.testutil import TINY_CONFIG
from treecat.testutil import tempdir
from treecat.training import train_ensemble
from treecat.training import train_model


@pytest.mark.parametrize('model_type', ['single', 'ensemble'])
def test_e2e(model_type):
    with tempdir() as dirname:
        data_csv = os.path.join(TESTDATA, 'tiny_data.csv')
        config = TINY_CONFIG.copy()

        print('Guess schema.')
        types_csv = os.path.join(dirname, 'types.csv')
        values_csv = os.path.join(dirname, 'values.csv')
        guess_schema(data_csv, types_csv, values_csv)

        print('Load schema')
        schema = load_schema(types_csv, values_csv)
        ragged_index = schema['ragged_index']

        print('Load data')
        data = load_data(schema, data_csv)
        dataset = {'schema': schema, 'data': data}

        print('Train model')
        if model_type == 'single':
            model = train_model(ragged_index, data, config)
        else:
            model = train_ensemble(ragged_index, data, config)

        print('Serve model')
        server = serve_model(dataset, model)

        print('Query model')
        evidence = {'genre': 'drama'}
        server.logprob([evidence])
        samples = server.sample(100)
        server.logprob(samples)
        samples = server.sample(100, evidence)
        server.logprob(samples)
        try:
            median = server.median([evidence])
            server.logprob(median)
        except NotImplementedError:
            pass

        print('Examine latent structure')
        server.latent_perplexity()
        try:
            server.latent_correlation()
        except NotImplementedError:
            pass
