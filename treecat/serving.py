from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy.stats import entropy

from treecat.structure import TreeStructure
from treecat.structure import make_propagation_schedule
from treecat.util import profile
from treecat.util import sample_from_probs

logger = logging.getLogger(__name__)


def correlation(probs):
    """Compute correlation rho(X,Y) = sqrt(1 - exp(-2 I(X;Y)))."""
    assert probs.shape[0] == probs.shape[1]
    mutual_information = (entropy(probs.sum(0)) + entropy(probs.sum(1)) -
                          entropy(probs.flatten()))
    return np.sqrt(1.0 - np.exp(-2.0 * mutual_information))


class TreeCatServer(object):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, tree, suffstats, config):
        logger.info('NumpyServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        self._tree = tree
        self._config = config
        self._ragged_index = suffstats['ragged_index']
        self._schedule = make_propagation_schedule(tree.tree_grid)
        self._zero_row = np.zeros(self._ragged_index[-1], np.int8)

        # These are useful dimensions to import into locals().
        V = self._tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        M = self._config[
            'model_num_clusters']  # Clusters in each mixture model.
        self._VEM = (V, E, M)

        # Use Jeffreys priors.
        vert_prior = 0.5
        edge_prior = 0.5 / M
        feat_prior = 0.5 / M
        self._vert_probs = suffstats['vert_ss'].astype(np.float32) + vert_prior
        self._edge_probs = suffstats['edge_ss'].astype(np.float32) + edge_prior
        self._feat_probs = suffstats['feat_ss'].astype(np.float32) + feat_prior

    def zero_row(self):
        """Make an empty data row."""
        return self._zero_row.copy()

    @profile
    def sample(self, counts, data=None):
        """Sample from the posterior conditional distribution.

        Let V be the number of features and N be the number of rows in input
        data. This function draws in parallel N samples, each sample
        conditioned on one of the input data rows.

        Args:
          counts: A [V] numpy array of requested counts of multinomials to
            sample.
          data: An optional single row of conditioning data, as a ragged nummpy
            array of multinomial counts.

        Returns:
          A single row of sampled data, as a length-V list of numpy arrays of
          sampled multinomial data.
        """
        logger.debug('sampling data')
        V, E, M = self._VEM
        if data is None:
            data = self._zero_row
        assert data.shape == self._zero_row.shape
        assert data.dtype == self._zero_row.dtype
        assert counts.shape == (V, )

        feat_probs = self._feat_probs
        edge_probs = self._edge_probs
        vert_probs = self._vert_probs
        messages = vert_probs.copy()
        vert_sample = np.zeros([V], np.int32)
        feat_sample = self._zero_row.copy()

        for op, v, v2, e in self._schedule:
            message = messages[v, :]
            if op == 0:  # OP_UP
                # Propagate upward from observed to latent.
                beg, end = self._ragged_index[v:v + 2]
                obs_lat = feat_probs[beg:end, :].copy()
                lat = obs_lat.sum(axis=0)
                for c, count in enumerate(data[beg:end]):
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
                    beg, end = self._ragged_index[v:v + 2]
                    probs = feat_probs[beg:end, vert_sample[v]].copy()
                    probs /= probs.sum()
                    feat_sample[beg:end] = np.random.multinomial(
                        counts[v], probs)

        return feat_sample

    @profile
    def logprob(self, data):
        """Compute non-normalized log probability of a row of data.

        Let V be the number of features and N be the number of rows in input
        data. This function computes in the logprob of each of the N input
        data rows.

        Args:
          data: A ragged nummpy array of multinomial count data.

        Returns:
          An [N] numpy array of log probabilities.
        """
        logger.debug('computing logprob')
        V, E, M = self._VEM
        assert data.shape == (self._ragged_index[-1], )

        feat_probs = self._feat_probs
        edge_probs = self._edge_probs
        vert_probs = self._vert_probs
        messages = vert_probs.copy()
        logprob = 0.0

        for op, v, v2, e in self._schedule:
            message = messages[v, :]
            if op == 0:  # OP_UP
                # Propagate upward from observed to latent.
                beg, end = self._ragged_index[v:v + 2]
                obs_lat = feat_probs[beg:end, :].copy()
                lat = obs_lat.sum(axis=0)
                for c, count in enumerate(data[beg:end]):
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

    @profile
    def correlation(self):
        """Compute correlation matrix among latent features.

        This computes the generalization of Pearson's correlation to discrete
        data. Let I(X;Y) be the mutual information. Then define correlation as

          rho(X,Y) = sqrt(1 - exp(-2 I(X;Y)))

        Returns:
          An [V, V] numpy array of feature-feature correlations.
        """
        logger.debug('computing correlation')
        V, E, M = self._VEM
        edge_probs = self._edge_probs
        vert_probs = self._vert_probs
        result = np.zeros([V, V], np.float32)
        for root in range(V):
            messages = np.empty([V, M, M])
            schedule = make_propagation_schedule(self._tree.tree_grid, root)
            for op, v, v2, e in schedule:
                if op == 2:  # OP_ROOT
                    messages[v, :, :] = np.diagflat(vert_probs[v, :])
                elif op == 3:  # OP_OUT
                    trans = edge_probs[e, :, :]
                    if v > v2:
                        trans = trans.T
                    messages[v, :, :] = np.dot(
                        trans / vert_probs[v2, np.newaxis, :],
                        messages[v2, :, :])
            for v in range(V):
                result[root, v] = correlation(messages[v, :, :])
        return result


def serve_model(tree, suffstats, config):
    return TreeCatServer(tree, suffstats, config)
