from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from treecat.structure import make_propagation_schedule
from treecat.util import profile

logger = logging.getLogger(__name__)


class NumpyTrainer(object):
    def __init__(self, data, mask, config):
        logger.info('NumpyTrainer of %d x %d data', data.shape[0],
                    data.shape[1])
        super(NumpyTrainer, self).__init__(data, mask, config)
        self._init_tensors()
        self._schedule = make_propagation_schedule(self.tree.tree_grid)

        # These are useful dimensions to import into locals().
        V = self.tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        K = V * (V - 1) // 2  # Number of edges in the complete graph.
        M = self._config['num_clusters']  # Clusters in each mixture model.
        C = self._config['num_categories']  # Categories for each feature.
        self._VEKMC = (V, E, K, M, C)

    def _init_tensors(self):
        V, E, K, M, C = self._VEKMC

        # Hard-code these hyperparameters.
        feat_prior = 0.5  # Jeffreys prior.
        vert_prior = 1.0 / M  # Nonparametric.
        edge_prior = 1.0 / M**2  # Nonparametric.

        # Sufficient statistics are maintained always.
        self._vert_ss = np.zeros([V, M], np.int32)
        self._edge_ss = np.zeros([E, M, M], np.int32)
        self._feat_ss = np.zeros([V, C, M], np.int32)

        # Sufficient statistics for tree learning are reset after each batch.
        # This is the most expensive data structure, costing O(V^2 M^2) space.
        self._tree_ss = np.zeros([K, M, M], np.int32)
        self._tree_ss += edge_prior

        # Non-normalized probabilities are maintained within each batch.
        self._feat_probs = feat_prior + self._feat_ss.astype(np.float32)
        self._vert_probs = vert_prior + self._vert_ss.astype(np.float32)
        self._edge_probs = edge_prior + self._edge_ss.astype(np.float32)

        # Temporaries.
        self._messages = np.zeros([V, M], dtype=np.float32)

    @profile
    def _update_tensors(self, data, mask, assignments, diff):
        V, E, K, M, C = self._VEKMC

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
    def add_row(self, data, mask, assignments_out):
        logger.debug('add_row')
        V, E, K, M, C = self._VEKMC
        assert data.dtype == np.int32
        assert data.shape == (V, )
        assert mask.dtype == np.bool_
        assert mask.shape == (V, )
        assert assignments_out.dtype == np.int32_
        assert assignments_out.shape == (V, )

        tree = self._tree
        feat_probs = self._feat_probs
        feat_probs_sum = self._feat_probs.sum(axis=1)  # TODO optimize.
        vert_probs = self._vert_probs
        edge_probs = self._edge_probs
        messages = self._messages

        for v, parent, children in reversed(self._schedule):
            message = messages[v, :]
            message[:] = vert_probs[v, :]
            # Propagate upward from observed to latent.
            if mask[v]:
                message *= feat_probs[v, data[v], :]
                message /= feat_probs_sum[v, :]
            # Propagate latent state inward from children to v.
            for child in children:
                e = tree.find_edge(v, child)
                if v < child:
                    trans = edge_probs[e, :, :]
                else:
                    trans = np.transpose(edge_probs[e, :, :])
                message *= np.dot(trans, messages[child, :])
                message /= vert_probs[v, :]
            message /= np.reduce_max(message)

        # Propagate latent state outward from parent to v.
        for v, parent, children in self._schedule:
            message = messages[v, :]
            if parent is not None:
                e = tree.find_edge(v, parent)
                if parent < v:
                    trans = edge_probs[e, :, :]
                else:
                    trans = np.transpose(edge_probs[e, :, :])
                message *= trans[assignments_out[parent], :]
                message /= vert_probs[v, :]
                assert message.shape == (M, )
            message /= message.sum()
            assignments_out[v] = np.random.choice(M, p=message)
        self._update_tensors(data, mask, assignments_out, +1)

    def remove_row(self, data, mask, assignments_in):
        logger.debug('remove_row')
        V, E, K, M, C = self._VEKMC
        assert data.dtype == np.int32
        assert data.shape == (V, )
        assert mask.dtype == np.bool_
        assert mask.shape == (V, )
        assert assignments_in.dtype == np.int32_
        assert assignments_in.shape == (V, )
        self._update_tensors(data, mask, assignments_in, -1)
