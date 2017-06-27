from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from treecat.util import COUNTERS
from treecat.util import sizeof

logger = logging.getLogger(__name__)


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

    def entropy(self, feature_sets, cond_data=None, cond_mask=None):
        """Compute entropy of a subset of features, conditioned on data.

        Args:
          feature_sets: A list of sets of fetaures on which to compute entropy.
          cond_data: An optional row of data on which to condition.
          cond_mask: An optional row presence/absence values for cond_row.
            This defaults to all-false and hence unconditioned entropy.

        Returns:
          A numpy array of entropy estimates, one estimate per feature set.
        """
        logger.info('Computing entropy of %d feature sets', len(feature_sets))
        assert isinstance(feature_sets, list)
        assert (cond_data is None) == (cond_mask is None)
        N = self._config['serving_samples']
        V = self._tree.num_vertices
        for feature_set in feature_sets:
            for feature in feature_set:
                assert 0 <= feature and feature <= V
        data = np.zeros([N, V], dtype=np.int32)
        mask = np.zeros([V], dtype=np.int32)

        # Generate samples.
        if cond_data is not None and cond_mask is not None:
            assert cond_data.shape == cond_mask.shape
            assert cond_data.dtype == np.int32
            assert cond_mask.dtype == np.bool_
            data[...] = cond_data[np.newaxis, :]
            mask[:] = cond_mask
        samples = self.sample(data, mask)
        assert samples.shape == (N, V)

        # Compte entropies.
        result = np.zeros([len(feature_sets)], dtype=np.float32)
        for i, feature_set in enumerate(feature_sets):
            mask[:] = False
            mask[list(feature_set)] = True
            logprob = self.logprob(samples, mask)
            result[i] = -np.mean(logprob * np.exp(logprob))

        return result

    def correlation(self):
        """Computes correlation matrix among all features.

        Returns: A symmetrix matrix with entries
          rho(X,Y) = sqrt(1 - exp(-2 I(X;Y)))
        """
        logger.info('Computing among %d feature sets', self._tree.num_vertices)

        # Compute entropy H(X) of features and H(X,Y) of pairs of features.
        V = self._tree.num_vertices
        feature_sets = []
        for v1 in range(V):
            feature_sets.append([v1])
            for v2 in range(v1):
                feature_sets.append([v1, v2])
        entropies = self.entropy(feature_sets)

        # Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
        mutual_information = np.zeros([V, V], np.float32)
        for feature_set, entropy in zip(feature_sets, entropies):
            v1 = min(feature_set)
            v2 = max(feature_set)
            mutual_information[v1, v2] -= entropy
            mutual_information[v2, v1] -= entropy
            if v1 == v2:
                mutual_information[v1, :] += entropy
                mutual_information[:, v2] += entropy

        # Copute correlation rho(X,Y) = sqrt(1 - exp(-2 I(X;Y))).
        return np.sqrt(1.0 - np.exp(-2.0 * mutual_information))


def serve_model(tree, suffstats, config):
    if config['engine'] == 'numpy':
        from treecat.np_engine import NumpyServer as Server
    else:
        raise ValueError('Unknown engine: {}'.format(config['engine']))
    return Server(tree, suffstats, config)
