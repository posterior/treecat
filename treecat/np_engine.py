from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy.special import gammaln

from treecat.structure import make_propagation_schedule
from treecat.structure import sample_tree
from treecat.training import TrainerBase
from treecat.util import COUNTERS
from treecat.util import profile
from treecat.util import sizeof

logger = logging.getLogger(__name__)


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

        vertices = np.arange(V, dtype=np.int32)
        tree_edges = np.arange(E, dtype=np.int32)
        complete_edges = np.arange(K, dtype=np.int32)
        data_mask = data[mask]
        assignments_mask = assignments[mask]
        assignments_e = (assignments[self.tree.tree_grid[1, :]],
                         assignments[self.tree.tree_grid[2, :]])
        self._feat_ss[mask, data_mask, assignments_mask] += 1
        self._vert_ss[vertices, assignments] += 1
        self._edge_ss[tree_edges, assignments_e[0], assignments_e[1]] += 1
        if diff > 0:
            self._tree_ss[complete_edges,  #
                          assignments[self.tree.complete_grid[1, :]],  #
                          assignments[self.tree.complete_grid[2, :]]] += 1

    @profile
    def add_row(self, row_id):
        logger.debug('NumpyTrainer.add_row %d', row_id)
        V, E, K, M, C = self._VEKMC
        feat_probs = self._feat_probs
        feat_probs_sum = self._feat_probs.sum(axis=1)  # TODO optimize.
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
                e = self.tree.find_edge(v, child)
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
                e = self.tree.find_edge(v, parent)
                if parent < v:
                    trans = edge_probs[e, :, :]
                else:
                    trans = np.transpose(edge_probs[e, :, :])
                message *= trans[assignments[parent], :]
                message /= vert_probs[v, :]
                assert message.shape == (M, )
            message /= message.sum()
            assignments[v] = np.random.choice(M, p=message)
        self._assigned_rows.add(row_id)
        self._update_tensors(row_id, +1)

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
        for k in range(K):
            block = self._tree_ss[k, :, :].astype(np.float32)
            block += self._edge_prior
            edge_logits[k] = gammaln(block).sum()

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
