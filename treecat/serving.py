from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy.misc import logsumexp
from scipy.stats import entropy

from treecat.structure import TreeStructure
from treecat.structure import estimate_tree
from treecat.structure import make_propagation_program
from treecat.util import profile
from treecat.util import sample_from_probs2

logger = logging.getLogger(__name__)


def correlation(probs):
    """Compute correlation rho(X,Y) = sqrt(1 - exp(-2 I(X;Y))).

    Args:
      probs: An [M, M]-shaped numpy array representing a joint distribution.

    Returns:
      A number in [0,1) representing the information-theoretic correlation.
    """
    assert len(probs.shape) == 2
    assert probs.shape[0] == probs.shape[1]
    mutual_information = (entropy(probs.sum(0)) + entropy(probs.sum(1)) -
                          entropy(probs.flatten()))
    return np.sqrt(1.0 - np.exp(-2.0 * mutual_information))


class ServerBase(object):
    """Base class for TreeCat and Ensemble servers."""

    def __init__(self, ragged_index):
        self._ragged_index = ragged_index
        self._zero_row = np.zeros(ragged_index[-1], np.int8)

    @property
    def ragged_index(self):
        return self._ragged_index

    def make_zero_row(self):
        """Make an empty data row."""
        return self._zero_row.copy()


class TreeCatServer(ServerBase):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, model):
        """Create a TreeCat server.

        Args:
          model: A dict with fields:
            tree: A TreeStructure.
            suffstats: A dict of sufficient statistics.
            edge_logits: A K-sized array of nonnormalized edge probabilities.
            config: A global config dict.
        """
        tree = model['tree']
        suffstats = model['suffstats']
        config = model['config']
        logger.info('TreeCatServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        ragged_index = suffstats['ragged_index']
        ServerBase.__init__(self, ragged_index)
        self._tree = tree
        self._config = config
        self._program = make_propagation_program(tree.tree_grid)

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
        meas_prior = feat_prior * np.array(
            [(ragged_index[v + 1] - ragged_index[v]) for v in range(V)],
            dtype=np.float32).reshape((V, 1))

        # These are posterior marginals for vertices and pairs of vertices.
        self._vert_probs = suffstats['vert_ss'].astype(np.float32) + vert_prior
        self._vert_probs /= self._vert_probs.sum(axis=1, keepdims=True)
        self._edge_probs = suffstats['edge_ss'].astype(np.float32) + edge_prior
        self._edge_probs /= self._edge_probs.sum(axis=(1, 2), keepdims=True)

        # This represents information in the pairwise joint posterior minus
        # information in the individual factors.
        self._edge_trans = self._edge_probs.copy()
        for e, v1, v2 in tree.tree_grid.T:
            self._edge_trans[e, :, :] /= self._vert_probs[v1, np.newaxis, :]
            self._edge_trans[e, :, :] /= self._vert_probs[v2, :, np.newaxis]

        # This is the conditional distribution of features given latent.
        self._feat_cond = suffstats['feat_ss'].astype(np.float32) + feat_prior
        meas_probs = suffstats['meas_ss'].astype(np.float32) + meas_prior
        for v in range(V):
            beg, end = ragged_index[v:v + 2]
            self._feat_cond[beg:end, :] /= meas_probs[v, np.newaxis, :]

        # These are used to inspect and visualize latent structure.
        self._edge_logits = model['edge_logits']
        self._estimated_tree = tuple(
            estimate_tree(self._tree.complete_grid, self._edge_logits))
        self._tree.gc()

    @property
    def edge_logits(self):
        return self._edge_logits

    @property
    def estimate_tree(self):
        return self._estimated_tree

    @profile
    def sample(self, N, counts, data=None):
        """Draw N samples from the posterior distribution.

        Args:
          size: The number of samples to draw.
          counts: A [V]-shaped numpy array of requested counts of multinomials
            to sample.
          data: An optional single row of conditioning data, as a ragged nummpy
            array of multinomial counts.

        Returns:
          An [N, _]-shaped numpy array of sampled multinomial data.
        """
        logger.debug('sampling data')
        V, E, M = self._VEM
        if data is None:
            data = self._zero_row
        assert data.shape == self._zero_row.shape
        assert data.dtype == self._zero_row.dtype
        assert counts.shape == (V, )
        assert counts.dtype == np.int8
        edge_trans = self._edge_trans
        feat_cond = self._feat_cond

        messages_in = self._vert_probs.copy()
        messages_out = np.tile(self._vert_probs[:, np.newaxis, :], (1, N, 1))
        vert_samples = np.zeros([V, N], np.int8)
        feat_samples = np.zeros([N, self._zero_row.shape[0]], np.int8)
        range_N = np.arange(N, dtype=np.int32)

        for op, v, v2, e in self._program:
            if op == 0:  # OP_UP
                # Propagate upward from observed to latent.
                message = messages_in[v, :]
                beg, end = self._ragged_index[v:v + 2]
                for r in range(beg, end):
                    # This uses a with-replacement approximation which is exact
                    # for categorical data but approximate for multinomial.
                    message *= feat_cond[r, :]**data[r]
            elif op == 1:  # OP_IN
                # Propagate latent state inward from children to v.
                message = messages_in[v, :]
                trans = edge_trans[e, :, :]
                if v > v2:
                    trans = trans.T
                message *= np.dot(trans, messages_in[v2, :])
                message /= message.sum()
            else:  # OP_ROOT or OP_OUT
                message = messages_out[v, :, :]
                message[...] = messages_in[v, np.newaxis, :]
                # Propagate latent state outward from parent to v.
                if op == 3:  # OP_OUT
                    trans = edge_trans[e, :, :]
                    if v2 > v:
                        trans = trans.T
                    message *= trans[vert_samples[v2, :], :]
                message /= message.sum(axis=1, keepdims=True)
                vert_samples[v, :] = sample_from_probs2(message)
                # Propagate downward from latent to observed.
                beg, end = self._ragged_index[v:v + 2]
                feat_block = feat_cond[beg:end, :].T
                probs = feat_block[vert_samples[v, :], :]
                samples_block = feat_samples[:, beg:end]
                for _ in range(counts[v]):
                    samples_block[range_N, sample_from_probs2(probs)] += 1

        return feat_samples

    @profile
    def logprob(self, data):
        """Compute non-normalized log probabilies of many rows of data.

        To compute conditional probabilty, use the identity:

          log P(data|cond_data) = server.logprob(data + cond_data)
                                - server.logprob(cond_data)

        Args:
          data: A [N, _]-shaped ragged nummpy array of multinomial count data,
            where N is the number of rows.

        Returns:
          An [N]-shaped numpy array of log probabilities.
        """
        logger.debug('computing logprob')
        assert len(data.shape) == 2
        assert data.shape[1] == self._ragged_index[-1]
        assert data.dtype == np.int8
        N = data.shape[0]
        V, E, M = self._VEM
        edge_trans = self._edge_trans
        feat_cond = self._feat_cond

        messages = np.tile(self._vert_probs[:, :, np.newaxis], (1, 1, N))
        assert messages.shape == (V, M, N)
        logprob = np.zeros(N, np.float32)

        for op, v, v2, e in self._program:
            message = messages[v, :, :]
            if op == 0:  # OP_UP
                # Propagate upward from observed to latent.
                beg, end = self._ragged_index[v:v + 2]
                for r in range(beg, end):
                    # This uses a with-replacement approximation which is exact
                    # for categorical data but approximate for multinomial.
                    power = data[np.newaxis, :, r]
                    message *= feat_cond[r, :, np.newaxis]**power
            elif op == 1:  # OP_IN
                # Propagate latent state inward from children to v.
                trans = edge_trans[e, :, :]
                if v > v2:
                    trans = trans.T
                message *= np.dot(trans, messages[v2, :, :])
                message_sum = message.sum(axis=0, keepdims=True)
                message /= message_sum
                logprob += np.log(message_sum[0, :])
            elif op == 2:  # OP_ROOT
                # Aggregate the total logprob.
                logprob += np.log(message.sum(axis=0))
                return logprob

    @profile
    def latent_correlation(self):
        """Compute correlation matrix among latent features.

        This computes the generalization of Pearson's correlation to discrete
        data. Let I(X;Y) be the mutual information. Then define correlation as

          rho(X,Y) = sqrt(1 - exp(-2 I(X;Y)))

        Returns:
          An [V, V] numpy array of feature-feature correlations.
        """
        logger.debug('computing latent correlation')
        V, E, M = self._VEM
        edge_probs = self._edge_probs
        vert_probs = self._vert_probs
        result = np.zeros([V, V], np.float32)
        for root in range(V):
            messages = np.empty([V, M, M])
            program = make_propagation_program(self._tree.tree_grid, root)
            for op, v, v2, e in program:
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


class EnsembleServer(ServerBase):
    """Class for serving queries against a trained TreeCat ensemble model."""

    def __init__(self, ensemble):
        logger.info('EnsembleServer of size %d', len(ensemble))
        assert ensemble
        ServerBase.__init__(self, ensemble[0]['suffstats']['ragged_index'])
        self._ensemble = [TreeCatServer(model) for model in ensemble]

        # These are used to inspect and visualize latent structure.
        self._edge_logits = self._ensemble[0].edge_logits.copy()
        for server in self._ensemble[1:]:
            self._edge_logits += server.edge_logits
        self._edge_logits /= len(self._ensemble)
        grid = self._ensemble[0]._tree.complete_grid
        self._estimated_tree = tuple(estimate_tree(grid, self._edge_logits))
        self._ensemble[0]._tree.gc()

    @property
    def edge_logits(self):
        return self._edge_logits

    @property
    def estimate_tree(self):
        return self._estimated_tree

    def sample(self, N, counts, data=None):
        size = len(self._ensemble)
        pvals = np.ones(size, dtype=np.float32) / size
        sub_Ns = np.random.multinomial(N, pvals)
        samples = np.concatenate([
            server.sample(sub_N, counts, data)
            for server, sub_N in zip(self._ensemble, sub_Ns)
        ])
        np.random.shuffle(samples)
        assert samples.shape[0] == N
        return samples

    def logprob(self, data):
        logprobs = np.stack(
            [server.logprob(data) for server in self._ensemble])
        logprobs = logsumexp(logprobs, axis=0)
        logprobs -= np.log(len(self._ensemble))
        assert logprobs.shape == (data.shape[0], )
        return logprobs
