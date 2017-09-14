from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from warnings import warn

import matplotlib
import pytest

from treecat.format import guess_schema
from treecat.format import load_data
from treecat.format import load_schema
from treecat.serving import serve_model
from treecat.tables import TY_MULTINOMIAL
from treecat.tables import Table
from treecat.testutil import TESTDATA
from treecat.testutil import TINY_CONFIG
from treecat.testutil import tempdir
from treecat.training import train_ensemble
from treecat.training import train_model

# The Agg backend is required for headless testing.
matplotlib.use('Agg')
from treecat.plotting import plot_circular  # noqa: E402 isort:skip


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
        groups_csv = os.path.join(TESTDATA, 'tiny_groups.csv')
        schema = load_schema(types_csv, values_csv, groups_csv)
        ragged_index = schema['ragged_index']
        tree_prior = schema['tree_prior']

        print('Load data')
        data = load_data(schema, data_csv)
        feature_types = [TY_MULTINOMIAL] * len(schema['feature_names'])
        table = Table(feature_types, ragged_index, data)
        dataset = {
            'schema': schema,
            'data': data,  # DEPRECATED
            'table': table,
        }

        print('Train model')
        if model_type == 'single':
            model = train_model(table, tree_prior, config)
        elif model_type == 'ensemble':
            model = train_ensemble(table, tree_prior, config)
        else:
            raise ValueError(model_type)

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
            warn('{} median not implemented'.format(model_type))
            pass
        try:
            mode = server.mode([evidence])
            server.logprob(mode)
        except NotImplementedError:
            warn('{} mode not implemented'.format(model_type))
            pass

        print('Examine latent structure')
        server.feature_density()
        server.observed_perplexity()
        server.latent_perplexity()
        server.latent_correlation()
        server.estimate_tree()
        server.sample_tree(10)

        print('Plotting latent structure')
        plot_circular(server)
