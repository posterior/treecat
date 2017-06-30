from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from treecat.structure import TreeStructure
from treecat.structure import make_propagation_schedule
from treecat.util import profile

logger = logging.getLogger(__name__)


def sample_from_probs(probs, out):
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
    V, M = suffstats['vert_ss'].shape
    E = V - 1
    assert grid.shape == (3, E)
    assert suffstats['vert_ss'].shape == (V, M)
    assert suffstats['edge_ss'].shape == (E, M, M)
    assert len(suffstats['feat_ss']) == V
    for v in range(V):
        assert suffstats['feat_ss'][v].shape[1] == M

    # Use Jeffreys priors.
    vert_prior = 0.5
    edge_prior = 0.5 / M
    feat_prior = 0.5 / M

    # First compute overlapping joint posteriors.
    latent = vert_prior + suffstats['vert_ss'].astype(np.float32)
    latent_latent = edge_prior + suffstats['edge_ss'].astype(np.float32)
    latent /= latent.sum(axis=1, keepdims=True)
    latent_latent /= latent_latent.sum(axis=(1, 2), keepdims=True)
    observed_latent = [None] * V
    observed = [None] * V
    for v in range(V):
        observed_latent[v] = (
            feat_prior + suffstats['feat_ss'][v].astype(np.float32))

        # Correct observed_latent for partially observed data, so that its
        # latent marginals agree with latent. The observed marginal will
        # reflect present data plus expected imputed data.
        observed_latent[v] *= (latent[v, np.newaxis, :] /
                               observed_latent[v].sum(axis=0, keepdims=True))
        observed[v] = observed_latent[v].sum(axis=1)

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
    factors['latent_latent'] /= factors['latent'][grid[1, :], :, np.newaxis]
    factors['latent_latent'] /= factors['latent'][grid[2, :], np.newaxis, :]
    for v in range(len(factors['observed'])):
        factors['observed_latent'][v] /= factors['observed'][v][:, np.newaxis]
        factors['observed_latent'][v] /= factors['latent'][v][np.newaxis, :]

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
        self._VEM = (V, E, M)

    @profile
    def _propagate(self, task, data, counts=None):
        V, E, M = self._VEM
        factor_observed = self._factors['observed']
        factor_observed_latent = self._factors['observed_latent']
        factor_latent = self._factors['latent']
        factor_latent_latent = self._factors['latent_latent']

        messages = factor_latent.copy()
        logprob = 0.0
        for v, parent, children in reversed(self._schedule):
            message = messages[v, :]
            # Propagate upward from observed to latent.
            for c, count in enumerate(data[v]):
                if count:
                    message[:] *= factor_observed_latent[v][c, :] ** count
                    logprob += np.log(factor_observed[v][c]) * count
            # Propagate latent state inward from children to v.
            for child in children:
                e = self._tree.find_tree_edge(child, v)
                trans = factor_latent_latent[e, :, :]
                if child > v:
                    trans = trans.T
                message *= np.dot(messages[child, :], trans)
            message_sum = message.sum()
            message /= message_sum
            logprob += np.log(message_sum)

        if task == 'logprob':
            # Aggregate the total logprob.
            root, parent, children = self._schedule[0]
            assert parent is None
            logprob += np.log(messages[root, :].sum())
            return logprob

        elif task == 'sample':
            latent_sample = np.zeros([V], np.int32)
            observed_sample = [np.zeros_like(col) for col in data]
            for v, parent, children in self._schedule:
                message = messages[v, :]
                # Propagate latent state outward from parent to v.
                if parent is not None:
                    e = self._tree.find_tree_edge(parent, v)
                    trans = factor_latent_latent[e, :, :]
                    if parent > v:
                        trans = trans.T
                    message *= trans[latent_sample[parent], :]
                sample_from_probs(message, out=latent_sample[v, :])
                # Propagate downward from latent to observed.
                if counts[v]:
                    probs = factor_observed_latent[v][:, latent_sample[v]]
                    assert probs.shape == data[v].shape
                    probs /= probs.sum()
                    observed_sample[v][:] = np.random.multinomial(
                        counts[v], probs)
            return observed_sample

        raise ValueError('Unknown task: {}'.format(task))

    def sample(self, cond_data, counts):
        """Sample from the posterior conditional distribution.

        Let V be the number of features and N be the number of rows in input
        data. This function draws in parallel N samples, each sample
        conditioned on one of the input data rows.

        Args:
          cond_data: An length-V list of numpy arrays of multinomial data on
            which to condition.
            To sample from the unconditional posterior, set all data to zero.
          counts: A [V] numpy array of requested counts of multinomials to
            sample.

        Returns:
          An length-V list of numpy arrays of sampled multinomial data, where
          the cells that where conditioned on should match the value of input
          data, and the cells that were not present should be randomly sampled
          from the conditional posterior.
        """
        logger.info('sampling %d rows', cond_data[0].shape[0])
        V, E, M = self._VEM
        assert len(cond_data) == V
        assert counts.shape == (V, )
        return self._propagate('sample', cond_data, counts)

    def logprob(self, data):
        """Compute log probability of a row of data.

        Let V be the number of features and N be the number of rows in input
        data. This function computes in the logprob of each of the N input
        data rows.

        Args:
          data: An length-V list of numpy arrays of multinomial count data.

        Returns:
          An [N] numpy array of log probabilities.
        """
        logger.debug('computing logprob of %d rows', data.shape[0])
        N = data[0].shape[0]
        V, E, M = self._VEM
        assert len(data) == V
        for v in range(V):
            assert data[v].shape[0] == N
        return self._propagate('logprob', data)


def serve_model(tree, suffstats, config):
    return TreeCatServer(tree, suffstats, config)
