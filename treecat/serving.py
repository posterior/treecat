from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def make_posterior_factors(grid, suffstats):
    """Computes a complete factorization of the posterior.

    Args:
      suffstats: A dictionary with numpy arrays for sufficient statistics:
        vert_ss, edge_ss, and feat_ss.

    Returns:
      A dictionary with numpy arrays for factors of the posterior:
      observed, observed_latent, latent, latent_latent.
    """
    V, C, M = suffstats['feat_ss'].shape
    E = V - 1
    assert grid.shape == (3, E)
    assert suffstats['vert_ss'].shape == (V, M)
    assert suffstats['edge_ss'].shape == (E, M, M)

    # Hard-code these hyperparameters.
    feat_prior = 0.5  # Jeffreys prior.
    vert_prior = 1.0 / M  # Nonparametric.
    edge_prior = 1.0 / M**2  # Nonparametric.

    # First compute overlapping joint posteriors.
    latent = vert_prior + suffstats['vert_ss'].astype(np.float32)
    latent_latent = edge_prior + suffstats['edge_ss'].astype(np.float32)
    observed_latent = feat_prior + suffstats['feat_ss'].astype(np.float32)
    latent /= latent.sum(1, keepdims=True)
    latent_latent /= latent_latent.sum((1, 2), keepdims=True)
    observed_latent /= observed_latent.sum((1, 2), keepdims=True)

    # Correct observed_latent for partially observed data.
    partial_latent = observed_latent.sum(1)
    observed_latent *= (latent / partial_latent)[:, np.newaxis, :]

    # Finally compute a non-overlapping factorization.
    observed = observed_latent.sum(2)
    observed_latent /= observed[:, :, np.newaxis]
    observed_latent /= latent[:, np.newaxis, :]
    latent_latent /= latent[grid[1, :], :, np.newaxis]
    latent_latent /= latent[grid[2, :], np.newaxis, :]

    return {
        'observed': observed,
        'observed_latent': observed_latent,
        'latent': latent,
        'latent_latent': latent_latent,
    }


class ServerBase(object):
    """Base class for serving queries against a trained TreeCat model."""
    pass


def serve_model(tree, suffstats, config):
    if config['engine'] == 'tensorflow':
        from treecat.tf_engine import TensorflowServer as Server
    elif config['engine'] == 'numpy':
        from treecat.np_engine import NumpyServer as Server
    else:
        raise ValueError('Unknown engine: {}'.format(config['engine']))
    return Server(tree, suffstats, config)
