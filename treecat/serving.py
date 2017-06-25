from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from treecat.util import COUNTERS
from treecat.util import sizeof


def make_posterior(grid, suffstats):
    """Computes a posterior marginals.

    Args:
      grid: a 3 x E grid of (edge, vertex, vertex) triples.
      suffstats: A dictionary with numpy arrays for sufficient statistics:
        vert_ss, edge_ss, and feat_ss.

    Returns:
      A dictionary with numpy arrays for marginals of the posterior:
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
    latent /= latent.sum(axis=1, keepdims=True)
    latent_latent /= latent_latent.sum(axis=(1, 2), keepdims=True)
    observed_latent /= observed_latent.sum(axis=(1, 2), keepdims=True)

    # Correct observed_latent for partially observed data,
    # so that its marginals agree with observed and latent.
    partial_latent = observed_latent.sum(axis=1)
    observed_latent *= (latent / partial_latent)[:, np.newaxis, :]
    observed = observed_latent.sum(axis=2)

    COUNTERS.footprint_serving_observed = sizeof(observed)
    COUNTERS.footprint_serving_observed_latent = sizeof(observed_latent)
    COUNTERS.footprint_serving_latent = sizeof(latent)
    COUNTERS.footprint_serving_latent_latent = sizeof(latent_latent)

    return {
        'observed': observed,
        'observed_latent': observed_latent,
        'latent': latent,
        'latent_latent': latent_latent,
    }


def make_posterior_factors(grid, suffstats):
    """Computes a complete factorization of the posterior.

    Args:
      grid: a 3 x E grid of (edge, vertex, vertex) triples.
      suffstats: A dictionary with numpy arrays for sufficient statistics:
        vert_ss, edge_ss, and feat_ss.

    Returns:
      A dictionary with numpy arrays for factors of the posterior:
      observed, observed_latent, latent, latent_latent.
    """
    factors = make_posterior(grid, suffstats)

    # Remove duplicated information so that factorization is non-overlapping.
    factors['observed_latent'] /= factors['observed'][:, :, np.newaxis]
    factors['observed_latent'] /= factors['latent'][:, np.newaxis, :]
    factors['latent_latent'] /= factors['latent'][grid[1, :], :, np.newaxis]
    factors['latent_latent'] /= factors['latent'][grid[2, :], np.newaxis, :]

    return factors


class ServerBase(object):
    """Base class for serving queries against a trained TreeCat model."""

    def sample(self, data, mask):
        """Sample from the posterior conditional distribution.

        Let V be the number of features and N be the number of rows in input
        data. This function draws in parallel N samples, each sample
        conditioned on one of the input data rows.

        Args:
          data: An [N, V] numpy array of data on which to condition.
            Masked data values should be set to 0.
          mask: An [V] numpy array of presence/absence of conditioning data.
            The mask is constant across rows.
            To sample from the unconditional posterior, set mask to all False.

        Returns:
          An [N, V] numpy array of sampled data, where the cells that where
          conditioned on should match the value of input data, and the cells
          that were not present should be randomly sampled from the conditional
          posterior.
        """
        raise NotImplementedError()

    def logprob(self, data, mask):
        """Compute log probabilities of each row of data.

        Let V be the number of features and N be the number of rows in input
        data and mask. This function computes in parallel the logprob of each
        of the N input data rows.

        Args:
          data: An [N, V] numpy array of data on which to condition.
            Masked data values should be set to 0.
          mask: An [N, V] numpy array of presence/absence of conditioning data.

        Returns:
          An [N] numpy array of log probabilities.
        """
        raise NotImplementedError()


def serve_model(tree, suffstats, config):
    if config['engine'] == 'numpy':
        from treecat.np_engine import NumpyServer as Server
    elif config['engine'] == 'tensorflow':
        from treecat.tf_engine import TensorflowServer as Server
    elif config['engine'] == 'cython':
        from treecat.cy_engine import CythonServer as Server
    else:
        raise ValueError('Unknown engine: {}'.format(config['engine']))
    return Server(tree, suffstats, config)
