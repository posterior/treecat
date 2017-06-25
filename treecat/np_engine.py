from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy.special import gammaln

from six.moves import xrange
from treecat.serving import ServerBase
from treecat.serving import make_posterior_factors
from treecat.structure import TreeStructure
from treecat.structure import find_complete_edge
from treecat.structure import make_propagation_schedule
from treecat.structure import sample_tree
from treecat.training import TrainerBase
from treecat.util import COUNTERS
from treecat.util import profile
from treecat.util import sizeof

logger = logging.getLogger(__name__)


def sample_from_probs(probs):
    # Note: np.random.multinomial is faster than np.random.choice,
    # but np.random.multinomial is pickier about non-normalized probs.
    try:
        # This is equivalent to: np.random.choice(M, p=probs)
        return np.random.multinomial(1, probs).argmax()
    except ValueError:
        COUNTERS.np_random_multinomial_value_error += 1
        return probs.argmax()


def sample_from_probs2(probs, out):
    """Vectorized sampler from categorical distribution."""
    # Adapted from https://stackoverflow.com/questions/40474436
    assert len(probs.shape) == 2
    u = np.random.rand(probs.shape[0], 1)
    cdf = probs.cumsum(axis=1)
    (u < cdf).argmax(axis=1, out=out)


class NumpyTrainer(TrainerBase):
    """Class for training a TreeCat model using Numpy."""

    def __init__(self, data, mask, config):
        logger.info('NumpyTrainer of %d x %d data', data.shape[0],
                    data.shape[1])
        super(NumpyTrainer, self).__init__(data, mask, config)
        self._schedule = make_propagation_schedule(self.tree.tree_grid)

        # These are useful dimensions to import into locals().
        V = self.tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        K = V * (V - 1) // 2  # Number of edges in the complete graph.
        M = self._config['num_clusters']  # Clusters in each mixture model.
        C = self._config['num_categories']  # Categories for each feature.
        self._VEKMC = (V, E, K, M, C)

        # Hard-code these hyperparameters.
        self._feat_prior = 0.5  # Jeffreys prior.
        self._vert_prior = 1.0 / M  # Nonparametric.
        self._edge_prior = 1.0 / M**2  # Nonparametric.

        # Sufficient statistics are maintained always.
        self._vert_ss = np.zeros([V, M], np.int32)
        self._edge_ss = np.zeros([E, M, M], np.int32)
        self._feat_ss = np.zeros([V, C, M], np.int32)

        # Sufficient statistics for tree learning are reset after each batch.
        # This is the most expensive data structure, costing O(V^2 M^2) space.
        self._tree_ss = np.zeros([K, M, M], np.int32)

        # Non-normalized probabilities are maintained within each batch.
        self._feat_probs = self._feat_prior + self._feat_ss.astype(np.float32)
        self._vert_probs = self._vert_prior + self._vert_ss.astype(np.float32)
        self._edge_probs = self._edge_prior + self._edge_ss.astype(np.float32)

        # Temporaries.
        self._messages = np.zeros([V, M], dtype=np.float32)

        COUNTERS.footprint_training_vert_ss = sizeof(self._vert_ss)
        COUNTERS.footprint_training_edge_ss = sizeof(self._edge_ss)
        COUNTERS.footprint_training_feat_ss = sizeof(self._feat_ss)
        COUNTERS.footprint_training_tree_ss = sizeof(self._tree_ss)
        COUNTERS.footprint_training_vert_probs = sizeof(self._feat_probs)
        COUNTERS.footprint_training_vert_probs = sizeof(self._vert_probs)
        COUNTERS.footprint_training_edge_probs = sizeof(self._edge_probs)
        COUNTERS.footprint_training_messages = sizeof(self._messages)

    def _update_tree(self):
        for e, v1, v2 in self.tree.tree_grid.T:
            k = find_complete_edge(v1, v2)
            self._edge_ss[e, :, :] = self._tree_ss[k, :, :]
        self._schedule = make_propagation_schedule(self.tree.tree_grid)
        self._tree_ss[...] = 0
        self._feat_probs[...] = self._feat_ss
        self._feat_probs += self._feat_prior
        self._vert_probs[...] = self._vert_ss
        self._vert_probs += self._vert_prior
        self._edge_probs[...] = self._edge_ss
        self._edge_probs += self._edge_prior

    @profile
    def _update_tensors(self, row_id, diff):
        V, E, K, M, C = self._VEKMC
        data = self._data[row_id]
        mask = self._mask[row_id]
        assignments = self.assignments[row_id]

        vertices = self.tree.vertices
        tree_edges = self.tree.tree_grid[0, :]
        complete_edges = self.tree.complete_grid[0, :]
        data_mask = data[mask]
        assignments_mask = assignments[mask]
        assignments_e = (assignments[self.tree.tree_grid[1, :]],
                         assignments[self.tree.tree_grid[2, :]])
        self._feat_ss[mask, data_mask, assignments_mask] += 1
        self._vert_ss[vertices, assignments] += 1
        self._edge_ss[tree_edges, assignments_e[0], assignments_e[1]] += 1
        self._feat_probs[mask, data_mask, assignments_mask] += 1
        self._vert_probs[vertices, assignments] += 1
        self._edge_probs[tree_edges, assignments_e[0], assignments_e[1]] += 1
        if diff > 0:
            self._tree_ss[complete_edges,  #
                          assignments[self.tree.complete_grid[1, :]],  #
                          assignments[self.tree.complete_grid[2, :]]] += 1

    @profile
    def add_row(self, row_id):
        logger.debug('NumpyTrainer.add_row %d', row_id)
        V, E, K, M, C = self._VEKMC
        feat_probs = self._feat_probs
        feat_probs_sum = self._feat_probs.sum(axis=1)
        vert_probs = self._vert_probs
        edge_probs = self._edge_probs
        messages = self._messages
        data = self._data[row_id]
        mask = self._mask[row_id]
        assignments = self.assignments[row_id]

        for v, parent, children in reversed(self._schedule):
            message = messages[v, :]
            message[:] = vert_probs[v, :]
            # Propagate upward from observed to latent.
            if mask[v]:
                message *= feat_probs[v, data[v], :]
                message /= feat_probs_sum[v, :]
            # Propagate latent state inward from children to v.
            for child in children:
                e = self.tree.find_tree_edge(v, child)
                if v < child:
                    trans = edge_probs[e, :, :]
                else:
                    trans = np.transpose(edge_probs[e, :, :])
                message *= np.dot(trans, messages[child, :])
                message /= vert_probs[v, :]
            message /= np.max(message)

        # Propagate latent state outward from parent to v.
        for v, parent, children in self._schedule:
            message = messages[v, :]
            if parent is not None:
                e = self.tree.find_tree_edge(v, parent)
                if parent < v:
                    trans = edge_probs[e, :, :]
                else:
                    trans = np.transpose(edge_probs[e, :, :])
                message *= trans[assignments[parent], :]
                message /= vert_probs[v, :]
                assert message.shape == (M, )
            message /= message.sum()
            assignments[v] = sample_from_probs(message)
        self._assigned_rows.add(row_id)
        self._update_tensors(row_id, +1)

    @profile
    def remove_row(self, row_id):
        logger.debug('NumpyTrainer.remove_row %d', row_id)
        self._assigned_rows.remove(row_id)
        self._update_tensors(row_id, -1)

    @profile
    def sample_tree(self):
        logger.info('NumpyTrainer.sample_tree given %d rows',
                    len(self._assigned_rows))
        # Compute edge logits.
        V, E, K, M, C = self._VEKMC
        edge_logits = np.zeros([K], np.float32)
        block = np.zeros([M, M], dtype=np.float32)
        for k in xrange(K):
            block[...] = self._tree_ss[k, :, :]
            block += self._edge_prior
            edge_logits[k] = gammaln(block).sum()  # Costs ~8% of total time!

        # Sample the tree.
        complete_grid = self.tree.complete_grid
        assert edge_logits.shape[0] == complete_grid.shape[1]
        edges = self.tree.tree_grid[1:3, :].T
        edges = sample_tree(
            complete_grid,
            edge_logits,
            edges,
            seed=self._seed,
            steps=self._config['sample_tree_steps'])
        self._seed += 1
        self.tree.set_edges(edges)
        self._update_tree()

    def finish(self):
        logger.info('NumpyTrainer.finish with %d rows',
                    len(self._assigned_rows))
        self.suffstats = {
            'feat_ss': self._feat_ss,
            'vert_ss': self._vert_ss,
            'edge_ss': self._edge_ss,
        }
        self._tree_ss = None
        self.tree.gc()


class NumpyServer(ServerBase):
    def __init__(self, tree, suffstats, config):
        logger.info('TensorflowServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        self._tree = tree
        self._config = config
        self._seed = config['seed']
        self._factors = make_posterior_factors(tree.tree_grid, suffstats)
        self._schedule = make_propagation_schedule(tree.tree_grid)

        # These are useful dimensions to import into locals().
        V = self._tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        M = self._config['num_clusters']  # Clusters in each mixture model.
        C = self._config['num_categories']  # Categories for each feature.
        self._VEMC = (V, E, M, C)

    @profile
    def _propagate(self, data, mask, task):
        N = data.shape[0]
        V, E, M, C = self._VEMC
        factor_observed_latent = self._factors['observed_latent']
        factor_latent = self._factors['latent']
        factor_latent_latent = self._factors['latent_latent']

        messages = np.zeros([V, N, M], dtype=np.float32)
        messages[...] = factor_latent[:, np.newaxis, :]
        messages_scale = np.zeros([V, N], dtype=np.float32)
        for v, parent, children in reversed(self._schedule):
            message = messages[v, :, :]
            # Propagate upward from observed to latent.
            if mask[v]:
                message *= factor_observed_latent[v, data[:, v], :]
            # Propagate latent state inward from children to v.
            for child in children:
                e = self._tree.find_tree_edge(child, v)
                if child < v:
                    trans = factor_latent_latent[e, :, :]
                else:
                    trans = factor_latent_latent[e, :, :].T
                message *= np.dot(messages[child, :, :], trans)
            messages_scale[v, :] = message.max(axis=1)
            message /= messages_scale[v, :, np.newaxis]

        if task == 'logprob':
            # Aggregate the total logprob.
            root, parent, children = self._schedule[0]
            assert parent is None
            logprob = np.log(messages[root, :, :].sum(axis=1))
            logprob += np.log(messages_scale).sum(axis=0)
            return logprob

        elif task == 'sample':
            latent_samples = np.zeros([V, N], np.int32)
            observed_samples = data.copy()
            for v, parent, children in self._schedule:
                message = messages[v, :, :]
                # Propagate latent state outward from parent to v.
                if parent is not None:
                    e = self._tree.find_tree_edge(parent, v)
                    if parent < v:
                        trans = factor_latent_latent[e, :, :]
                    else:
                        trans = factor_latent_latent[e, :, :].T
                    message *= trans[latent_samples[parent, :], :]
                sample_from_probs2(message, out=latent_samples[v, :])
                # Propagate downward from latent to observed.
                if not mask[v]:
                    probs = factor_observed_latent[v, :, latent_samples[v, :]]
                    probs = probs.T / probs.sum()
                    sample_from_probs2(probs, out=observed_samples[:, v])
            return observed_samples

        raise ValueError('Unknown task: {}'.format(task))

    @profile
    def sample(self, data, mask):
        logger.info('sampling %d rows', data.shape[0])
        # TODO set seed
        N = data.shape[0]
        V, E, M, C = self._VEMC
        assert data.shape == (N, V)
        assert mask.shape == (V, )
        return self._propagate(data, mask, 'sample')

    @profile
    def logprob(self, data, mask):
        logger.info('computing logprob of %d rows', data.shape[0])
        N = data.shape[0]
        V, E, M, C = self._VEMC
        assert data.shape == (N, V)
        assert mask.shape == (V, )
        return self._propagate(data, mask, 'logprob')
