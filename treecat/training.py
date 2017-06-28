from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging

import numpy as np
from scipy.special import gammaln

from treecat.structure import TreeStructure
from treecat.structure import find_complete_edge
from treecat.structure import make_propagation_schedule
from treecat.structure import sample_tree
from treecat.util import COUNTERS
from treecat.util import art_logger
from treecat.util import profile
from treecat.util import sizeof

logger = logging.getLogger(__name__)


def logprob_dm(counts, prior):
    """Non-normalized log probability of a Dirichlet-Multinomial distribution.

    See https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    """
    return gammaln(counts + prior).sum() - gammaln(counts + 1.0).sum()


def sample_from_probs(probs):
    # Note: np.random.multinomial is faster than np.random.choice,
    # but np.random.multinomial is pickier about non-normalized probs.
    try:
        # This is equivalent to: np.random.choice(M, p=probs)
        return np.random.multinomial(1, probs).argmax()
    except ValueError:
        COUNTERS.np_random_multinomial_value_error += 1
        return probs.argmax()


class TreeCatTrainer(object):
    """Class for training a TreeCat model."""

    def __init__(self, data, mask, config):
        """Initialize a model in an unassigned state.

        Args:
            data: A 2D array of categorical data.
            mask: A 2D array of presence/absence, where present = True.
            config: A global config dict.
        """
        logger.info('TreeCatTrainer of %d x %d data', data.shape[0],
                    data.shape[1])
        data = np.asarray(data, np.int32)
        mask = np.asarray(mask, np.bool_)
        num_rows, num_features = data.shape
        assert data.shape == mask.shape
        self._data = data
        self._mask = mask
        self._config = config
        self._assigned_rows = set()
        self.assignments = np.zeros(data.shape, dtype=np.int32)
        self.suffstats = {}
        self.tree = TreeStructure(num_features)
        self._sampling_tree = (config['learning_sample_tree_steps'] > 0)
        self._schedule = make_propagation_schedule(self.tree.tree_grid)

        # These are useful dimensions to import into locals().
        V = self.tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        K = V * (V - 1) // 2  # Number of edges in the complete graph.
        M = self._config['model_num_clusters']  # Clusters per latent.
        C = self._config['model_num_categories']  # Categories per feature.
        self._VEKMC = (V, E, K, M, C)

        # Use Jeffreys priors.
        self._vert_prior = 0.5
        self._edge_prior = 0.5 / M
        self._feat_prior = 0.5 / M

        # Sufficient statistics are maintained always.
        self._vert_ss = np.zeros([V, M], np.int32)
        self._edge_ss = np.zeros([E, M, M], np.int32)
        self._feat_ss = np.zeros([V, C, M], np.int32)

        # Sufficient statistics for tree learning are reset after each batch.
        # This is the most expensive data structure, costing O(V^2 M^2) space.
        if self._sampling_tree:
            self._tree_ss = np.zeros([K, M, M], np.int32)
            COUNTERS.footprint_training_tree_ss = sizeof(self._tree_ss)

        # Temporaries.
        self._messages = np.zeros([V, M], dtype=np.float32)

        COUNTERS.footprint_training_data = sizeof(self._data)
        COUNTERS.footprint_training_mask = sizeof(self._mask)
        COUNTERS.footprint_training_assignments = sizeof(self.assignments)
        COUNTERS.footprint_training_vert_ss = sizeof(self._vert_ss)
        COUNTERS.footprint_training_edge_ss = sizeof(self._edge_ss)
        COUNTERS.footprint_training_feat_ss = sizeof(self._feat_ss)
        COUNTERS.footprint_training_messages = sizeof(self._messages)

    def _update_tree(self):
        assert self._sampling_tree
        for e, v1, v2 in self.tree.tree_grid.T:
            k = find_complete_edge(v1, v2)
            self._edge_ss[e, :, :] = self._tree_ss[k, :, :]
        self._schedule = make_propagation_schedule(self.tree.tree_grid)
        self._tree_ss[...] = 0

    @profile
    def _update_tensors(self, row_id, diff):
        V, E, K, M, C = self._VEKMC
        data = self._data[row_id]
        mask = self._mask[row_id]
        assignments = self.assignments[row_id]

        self._feat_ss[mask, data[mask], assignments[mask]] += diff
        self._vert_ss[self.tree.vertices, assignments] += diff
        self._edge_ss[self.tree.tree_grid[0, :],  #
                      assignments[self.tree.tree_grid[1, :]],  #
                      assignments[self.tree.tree_grid[2, :]]] += diff
        if self._sampling_tree and diff > 0:
            self._tree_ss[self.tree.complete_grid[0, :],  #
                          assignments[self.tree.complete_grid[1, :]],  #
                          assignments[self.tree.complete_grid[2, :]]] += diff

    @profile
    def add_row(self, row_id):
        logger.debug('TreeCatTrainer.add_row %d', row_id)
        assert row_id not in self._assigned_rows, row_id
        V, E, K, M, C = self._VEKMC
        messages = self._messages
        data = self._data[row_id]
        mask = self._mask[row_id]
        assignments = self.assignments[row_id]
        vert_probs = self._vert_ss + self._vert_prior

        for v, parent, children in reversed(self._schedule):
            message = messages[v, :]
            message[:] = vert_probs[v, :]
            # Propagate upward from observed to latent.
            if mask[v]:
                feat_probs = self._feat_ss[v, data[v], :] + self._feat_prior
                message *= feat_probs
                message /= feat_probs.sum(axis=0)
            # Propagate latent state inward from children to v.
            for child in children:
                e = self.tree.find_tree_edge(v, child)
                trans = self._edge_ss[e, :, :] + self._edge_prior
                if v > child:
                    trans = trans.T
                message *= np.dot(trans, messages[child, :])
                message /= vert_probs[v, :]
            message /= np.max(message)

        # Propagate latent state outward from parent to v.
        for v, parent, children in self._schedule:
            message = messages[v, :]
            if parent is not None:
                e = self.tree.find_tree_edge(v, parent)
                trans = self._edge_ss[e, :, :] + self._edge_prior
                if parent > v:
                    trans = trans.T
                message *= trans[assignments[parent], :]
                message /= vert_probs[v, :]
                assert message.shape == (M, )
            message /= message.sum()
            assignments[v] = sample_from_probs(message)
        self._assigned_rows.add(row_id)
        self._update_tensors(row_id, +1)

    @profile
    def remove_row(self, row_id):
        logger.debug('TreeCatTrainer.remove_row %d', row_id)
        assert row_id in self._assigned_rows, row_id
        self._assigned_rows.remove(row_id)
        self._update_tensors(row_id, -1)

    @profile
    def sample_tree(self):
        logger.info('TreeCatTrainer.sample_tree given %d rows',
                    len(self._assigned_rows))
        assert self._sampling_tree
        V, E, K, M, C = self._VEKMC
        # Compute vertex logits.
        vertex_logits = np.zeros([V], np.float32)
        for v in range(V):
            vertex_logits[v] = logprob_dm(self._vert_ss[v, :],
                                          self._vert_prior)
        # Compute edge logits.
        edge_logits = np.zeros([K], np.float32)
        for k, v1, v2 in self.tree.tree_grid.T:
            # This is the most expensive part of tree sampling:
            edge_logits[k] = (
                logprob_dm(self._tree_ss[k, :, :], self._edge_prior) -
                vertex_logits[v1] - vertex_logits[v2])

        # Sample the tree.
        complete_grid = self.tree.complete_grid
        assert edge_logits.shape[0] == complete_grid.shape[1]
        edges = self.tree.tree_grid[1:3, :].T
        edges = sample_tree(
            complete_grid,
            edge_logits,
            edges,
            steps=self._config['learning_sample_tree_steps'])
        self.tree.set_edges(edges)
        self._update_tree()

    def logprob(self):
        """Compute non-normalized log probability of data and assignments.

        This is mainly useful for testing goodness of fit of the category
        kernel.
        """
        # This uses inclusion-exclusion on the single and pairwise factors.
        V, E, K, M, C = self._VEKMC
        logprob = 0.0
        # Add contribution of each joint (observed, latent) distribution,
        # correcting for missing observations.
        for v in range(V):
            feat_block = self._feat_ss[v, :, :] + self._feat_prior
            feat_block *= (feat_block.sum(axis=0, keepdims=True) /
                           (self._vert_ss[v, :] + self._vert_prior))
            feat_block -= self._feat_prior
            logprob += logprob_dm(feat_block, self._feat_prior)
        # Keep track of logprobs of latent distribution for each vertex.
        logprobs = {}
        for v in range(V):
            logprobs[v] = logprob_dm(self._vert_ss[v, :], self._vert_prior)
        # Add contribution of each (latent, latent) joint distribution, and
        # remove the double-counted latent logprob of each of its vertices.
        for e, v1, v2 in self.tree.tree_grid.T:
            edge_block = self._edge_ss[e, :, :]
            logprob += (logprob_dm(edge_block, self._edge_prior) - logprobs[v1]
                        - logprobs[v2])
        return logprob

    def finish(self):
        logger.info('TreeCatTrainer.finish with %d rows',
                    len(self._assigned_rows))
        self.suffstats = {
            'feat_ss': self._feat_ss,
            'vert_ss': self._vert_ss,
            'edge_ss': self._edge_ss,
        }
        self._tree_ss = None
        self.tree.gc()

    def train(self):
        """Train a TreeCat model using subsample-annealed MCMC.

        Let N be the number of data rows and V be the number of features.

        Returns:
          A trained model as a dictionary with keys:
            tree: A TreeStructure instance with the learned latent structure.
            suffstats: Sufficient statistics of features, vertices, and
              edges.
            assignments: An [N, V] numpy array of latent cluster ids for each
              cell in the dataset.
        """
        logger.info('train()')
        np.random.seed(self._config['seed'])
        num_rows = self._data.shape[0]
        for action, row_id in get_annealing_schedule(num_rows, self._config):
            if action == 'add_row':
                art_logger('+')
                self.add_row(row_id)
            elif action == 'remove_row':
                art_logger('-')
                self.remove_row(row_id)
            else:
                art_logger('\n')
                self.sample_tree()
        self.finish()
        return {
            'config': self._config,
            'tree': self.tree,
            'suffstats': self.suffstats,
            'assignments': self.assignments,
        }


def train_model(data, mask, config):
    """Train a TreeCat model using subsample-annealed MCMC.

    Let N be the number of data rows and V be the number of features.

    Returns:
      A trained model as a dictionary with keys:
        tree: A TreeStructure instance with the learned latent structure.
        suffstats: Sufficient statistics of features, vertices, and
          edges.
        assignments: An [N, V] numpy array of latent cluster ids for each
          cell in the dataset.
    """
    return TreeCatTrainer(data, mask, config).train()


def get_annealing_schedule(num_rows, config):
    """Iterator for subsample annealing yielding (action, arg) pairs.

    Actions are one of: 'add_row', 'remove_row', or 'sample_tree'.
    The add and remove actions each provide a row_id arg.
    """
    # Randomly shuffle rows.
    row_ids = list(range(num_rows))
    np.random.shuffle(row_ids)
    row_to_add = itertools.cycle(row_ids)
    row_to_remove = itertools.cycle(row_ids)

    # Use a linear annealing schedule.
    epochs = float(config['learning_annealing_epochs'])
    add_rate = epochs
    remove_rate = epochs - 1.0
    state = epochs * config['learning_annealing_init_rows']

    # Sample the tree after each batch.
    sampling_tree = (config['learning_sample_tree_steps'] > 0)
    num_fresh = 0
    num_stale = 0
    while num_fresh + num_stale != num_rows:
        if state >= 0.0:
            yield 'add_row', next(row_to_add)
            state -= remove_rate
            num_fresh += 1
        else:
            yield 'remove_row', next(row_to_remove)
            state += add_rate
            num_stale -= 1
        if sampling_tree and num_stale == 0 and num_fresh > 0:
            yield 'sample_tree', None
            num_stale = num_fresh
            num_fresh = 0
