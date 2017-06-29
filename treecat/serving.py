from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from treecat.structure import TreeStructure
from treecat.structure import make_propagation_schedule
from treecat.util import profile

logger = logging.getLogger(__name__)


def sample_from_probs2(probs, out):
    """Vectorized sampler from categorical distribution."""
    # Adapted from https://stackoverflow.com/questions/40474436
    assert len(probs.shape) == 2
    u = np.random.rand(probs.shape[0], 1)
    cdf = probs.cumsum(axis=1)
    (u < cdf).argmax(axis=1, out=out)


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

    # Use Jeffreys priors.
    vert_prior = 0.5
    edge_prior = 0.5 / M
    feat_prior = 0.5 / M

    # First compute overlapping joint posteriors.
    latent = vert_prior + suffstats['vert_ss'].astype(np.float32)
    latent_latent = edge_prior + suffstats['edge_ss'].astype(np.float32)
    observed_latent = feat_prior + suffstats['feat_ss'].astype(np.float32)
    latent /= latent.sum(axis=1, keepdims=True)
    latent_latent /= latent_latent.sum(axis=(1, 2), keepdims=True)
    observed_latent /= observed_latent.sum(axis=(1, 2), keepdims=True)

    # Correct observed_latent for partially observed data, so that its
    # latent marginals agree with latent. The observed marginal will reflect
    # present data plus expected imputed data.
    partial_latent = observed_latent.sum(axis=1)
    observed_latent *= (latent / partial_latent)[:, np.newaxis, :]
    observed = observed_latent.sum(axis=2)

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


class TreeCatServer(object):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, tree, suffstats, config):
        logger.info('NumpyServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        self._tree = tree
        self._config = config
        self._factors = make_posterior_factors(tree.tree_grid, suffstats)
        self._schedule = make_propagation_schedule(tree.tree_grid)

        # These are useful dimensions to import into locals().
        V = self._tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        M = self._config[
            'model_num_clusters']  # Clusters in each mixture model.
        C = self._config[
            'model_num_categories']  # Categories for each feature.
        self._VEMC = (V, E, M, C)

    @profile
    def _propagate(self, data, mask, task):
        N = data.shape[0]
        V, E, M, C = self._VEMC
        factor_observed = self._factors['observed']
        factor_observed_latent = self._factors['observed_latent']
        factor_latent = self._factors['latent']
        factor_latent_latent = self._factors['latent_latent']

        messages = np.zeros([V, N, M], dtype=np.float32)
        messages[...] = factor_latent[:, np.newaxis, :]
        logprob = np.zeros([N], dtype=np.float32)
        for v, parent, children in reversed(self._schedule):
            message = messages[v, :, :]
            # Propagate upward from observed to latent.
            if mask[v]:
                message *= factor_observed_latent[v, data[:, v], :]
                logprob += np.log(factor_observed[v, data[:, v]])
            # Propagate latent state inward from children to v.
            for child in children:
                e = self._tree.find_tree_edge(child, v)
                trans = factor_latent_latent[e, :, :]
                if child > v:
                    trans = trans.T
                message *= np.dot(messages[child, :, :], trans)  # Expensive.
            message_scale = message.max(axis=1)  # Surprisingly expensive.
            message /= message_scale[:, np.newaxis]
            logprob += np.log(message_scale)

        if task == 'logprob':
            # Aggregate the total logprob.
            root, parent, children = self._schedule[0]
            assert parent is None
            logprob += np.log(messages[root, :, :].sum(axis=1))
            return logprob

        elif task == 'sample':
            latent_samples = np.zeros([V, N], np.int32)
            observed_samples = data.copy()
            for v, parent, children in self._schedule:
                message = messages[v, :, :]
                # Propagate latent state outward from parent to v.
                if parent is not None:
                    e = self._tree.find_tree_edge(parent, v)
                    trans = factor_latent_latent[e, :, :]
                    if parent > v:
                        trans = trans.T
                    message *= trans[latent_samples[parent, :], :]
                sample_from_probs2(message, out=latent_samples[v, :])
                # Propagate downward from latent to observed.
                if not mask[v]:
                    probs = factor_observed_latent[v, :, latent_samples[v, :]]
                    assert probs.shape == (N, C)
                    probs /= probs.sum(axis=1, keepdims=True)
                    sample_from_probs2(probs, out=observed_samples[:, v])
            return observed_samples

        raise ValueError('Unknown task: {}'.format(task))

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
        logger.info('sampling %d rows', data.shape[0])
        N = data.shape[0]
        V, E, M, C = self._VEMC
        assert data.shape == (N, V)
        assert mask.shape == (V, )
        return self._propagate(data, mask, 'sample')

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
        logger.debug('computing logprob of %d rows', data.shape[0])
        N = data.shape[0]
        V, E, M, C = self._VEMC
        assert data.shape == (N, V)
        assert mask.shape == (V, )
        return self._propagate(data, mask, 'logprob')

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

        # Compute correlation rho(X,Y) = sqrt(1 - exp(-2 I(X;Y))).
        return np.sqrt(1.0 - np.exp(-2.0 * mutual_information))


def serve_model(tree, suffstats, config):
    return TreeCatServer(tree, suffstats, config)
