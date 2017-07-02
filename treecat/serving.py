from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from treecat.structure import TreeStructure
from treecat.structure import make_propagation_schedule
from treecat.util import profile
from treecat.util import sample_from_probs

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


class TreeCatServer(object):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, tree, suffstats, config):
        logger.info('NumpyServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        self._tree = tree
        self._config = config
        self._posterior = make_posterior(tree.tree_grid, suffstats)
        self._schedule = make_propagation_schedule(tree.tree_grid)
        self._zero_row = [
            np.zeros(col.shape, dtype=np.int8)
            for col in self._posterior['observed']
        ]

        # These are useful dimensions to import into locals().
        V = self._tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        M = self._config[
            'model_num_clusters']  # Clusters in each mixture model.
        self._VEM = (V, E, M)

    def zero_row(self):
        """Make an empty data row."""
        return [col.copy() for col in self._zero_row]

    @profile
    def sample(self, data, counts):
        """Sample from the posterior conditional distribution.

        Let V be the number of features and N be the number of rows in input
        data. This function draws in parallel N samples, each sample
        conditioned on one of the input data rows.

        Args:
          data: A single row of conditioning data, as a length-V list of
            numpy arrays of multinomial data on which to condition.
            To sample from the unconditional posterior, set all data to zero.
          counts: A [V] numpy array of requested counts of multinomials to
            sample.

        Returns:
          A single row of sampled data, as a length-V list of numpy arrays of
          sampled multinomial data.
        """
        logger.debug('sampling %d rows', data[0].shape[0])
        V, E, M = self._VEM
        assert len(data) == V
        for v in range(V):
            assert data[v].shape == self._zero_row[v].shape
        assert counts.shape == (V, )

        feat_probs = self._posterior['observed_latent']
        edge_probs = self._posterior['latent_latent']
        vert_probs = self._posterior['latent']
        messages = vert_probs.copy()
        vert_sample = np.zeros([V], np.int32)
        feat_sample = [np.zeros_like(col) for col in data]

        for op, v, v2, e in self._schedule:
            message = messages[v, :]
            if op == 0:  # OP_UP
                # Propagate upward from observed to latent.
                obs_lat = feat_probs[v].copy()
                lat = obs_lat.sum(axis=0)
                for c, count in enumerate(data[v]):
                    for _ in range(count):
                        message *= obs_lat[c, :] / lat
                        obs_lat[c, :] += 1.0
                        lat += 1.0
            elif op == 1:  # OP_IN
                # Propagate latent state inward from v2ren to v.
                trans = edge_probs[e, :, :]
                if v > v2:
                    trans = trans.T
                message *= np.dot(trans, messages[v2, :] / vert_probs[v2, :])
                message /= vert_probs[v, :]
                message /= message.sum()
            else:  # OP_ROOT or OP_OUT
                # Propagate latent state outward from v2 to v.
                if op == 3:  # OP_OUT
                    trans = edge_probs[e, :, :]
                    if v2 > v:
                        trans = trans.T
                    message *= trans[vert_sample[v2], :]
                message /= message.sum()
                vert_sample[v] = sample_from_probs(message)
                # Propagate downward from latent to observed.
                if counts[v]:
                    probs = feat_probs[v][:, vert_sample[v]]
                    probs /= probs.sum()
                    feat_sample[v][:] = np.random.multinomial(counts[v], probs)

        return feat_sample

    @profile
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
        logger.debug('computing logprob')
        V, E, M = self._VEM
        assert len(data) == V
        for v in range(V):
            assert data[v].shape == self._zero_row[v].shape

        feat_probs = self._posterior['observed_latent']
        edge_probs = self._posterior['latent_latent']
        vert_probs = self._posterior['latent']
        messages = vert_probs.copy()
        logprob = 0.0

        for op, v, v2, e in self._schedule:
            message = messages[v, :]
            if op == 0:  # OP_UP
                # Propagate upward from observed to latent.
                obs_lat = feat_probs[v].copy()
                lat = obs_lat.sum(axis=0)
                for c, count in enumerate(data[v]):
                    for _ in range(count):
                        message *= obs_lat[c, :] / lat
                        obs_lat[c, :] += 1.0
                        lat += 1.0
            elif op == 1:  # OP_IN
                # Propagate latent state inward from v2ren to v.
                trans = edge_probs[e, :, :]
                if v > v2:
                    trans = trans.T
                message *= np.dot(trans, messages[v2, :] / vert_probs[v2, :])
                message /= vert_probs[v, :]
                message_sum = message.sum()
                message /= message_sum
                logprob += np.log(message_sum)
            elif op == 2:  # OP_ROOT
                # Aggregate the total logprob.
                logprob += np.log(message.sum())
                return logprob


def serve_model(tree, suffstats, config):
    return TreeCatServer(tree, suffstats, config)
