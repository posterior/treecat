from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging

import numpy as np
from scipy.special import gammaln

from six.moves import range
from treecat.structure import OP_IN
from treecat.structure import OP_OUT
from treecat.structure import OP_ROOT
from treecat.structure import OP_UP
from treecat.structure import TreeStructure
from treecat.structure import estimate_tree
from treecat.structure import make_propagation_program
from treecat.structure import sample_tree
from treecat.util import SQRT_TINY
from treecat.util import jit
from treecat.util import jit_sample_from_probs
from treecat.util import parallel_map
from treecat.util import profile
from treecat.util import set_random_seed

logger = logging.getLogger(__name__)


def count_pairs(assignments, v1, v2, M):
    """Construct sufficient statistics for (v1, v2) pairs.

    Args:
      assignments: An _ x V assignment matrix with values in range(M).
      v1, v2: Column ids of the assignments matrix.
      M: The number of possible assignment bins.

    Returns:
      And M x M array of counts.
    """
    assert v1 != v2
    pairs = assignments[:, v1].astype(np.int32) * M + assignments[:, v2]
    return np.bincount(pairs, minlength=M * M).reshape((M, M))


def logprob_dc(counts, prior, axis=None):
    """Non-normalized log probability of a Dirichlet-Categorical distribution.

    See https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    """
    # Not that this excludes the factorial(counts) term, since we explicitly
    # track permutations in assignments.
    return gammaln(np.add(counts, prior, dtype=np.float32)).sum(axis)


def get_annealing_schedule(num_rows, epochs):
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
    epochs = float(epochs)
    add_rate = epochs
    remove_rate = epochs - 1.0
    state = 2.0 * epochs

    # Sample the tree after each batch.
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
        if num_stale == 0 and num_fresh > 0:
            yield 'sample_tree', None
            num_stale = num_fresh
            num_fresh = 0


@profile
@jit(nopython=True, cache=True)
def jit_add_row(
        ragged_index,
        data_row,
        tree_grid,
        program,
        assignments,
        vert_ss,
        edge_ss,
        feat_ss,
        meas_ss,
        vert_probs,
        edge_probs,
        feat_probs,
        meas_probs, ):
    # Sample latent assignments using dynamic programming.
    messages = vert_probs.copy()
    for i in range(len(program)):
        op, v, v2, e = program[i]
        message = messages[v, :]
        if op == OP_UP:
            # Propagate upward from observed to latent.
            beg, end = ragged_index[v:v + 2]
            feat_block = feat_probs[beg:end, :]
            meas_block = meas_probs[v, :]
            for c, count in enumerate(data_row[beg:end]):
                for _ in range(count):
                    message *= feat_block[c, :]
                    message /= meas_block
                    feat_block[c, :] += np.float32(1)
                    meas_block += np.float32(1)
        elif op == OP_IN:
            # Propagate latent state inward from children to v.
            trans = edge_probs[e, :, :]
            if v > v2:
                trans = trans.T
            message *= np.dot(trans, messages[v2, :] / vert_probs[v2, :])
            message /= vert_probs[v, :]
            # Scale message for numerical stability.
            message /= message.max()
            message += SQRT_TINY
        elif op == OP_ROOT:
            # Process root node.
            assignments[v] = jit_sample_from_probs(message)
        elif op == OP_OUT:
            # Propagate latent state outward from parent to v.
            trans = edge_probs[e, :, :]
            if v2 > v:
                trans = trans.T
            message *= trans[assignments[v2], :]
            message /= vert_probs[v, :]
            assignments[v] = jit_sample_from_probs(message)

    # Update sufficient statistics.
    for v, m in enumerate(assignments):
        vert_ss[v, m] += 1
    for e in range(tree_grid.shape[1]):
        m1 = assignments[tree_grid[1, e]]
        m2 = assignments[tree_grid[2, e]]
        edge_ss[e, m1, m2] += 1
    for v, m in enumerate(assignments):
        beg, end = ragged_index[v:v + 2]
        feat_ss[beg:end, m] += data_row[beg:end]
        meas_ss[v, m] += data_row[beg:end].sum()


@profile
@jit(nopython=True, cache=True)
def jit_remove_row(
        ragged_index,
        data_row,
        tree_grid,
        assignments,
        vert_ss,
        edge_ss,
        feat_ss,
        meas_ss, ):
    # Update sufficient statistics.
    for v, m in enumerate(assignments):
        vert_ss[v, m] -= 1
    for e in range(tree_grid.shape[1]):
        m1 = assignments[tree_grid[1, e]]
        m2 = assignments[tree_grid[2, e]]
        edge_ss[e, m1, m2] -= 1
    for v, m in enumerate(assignments):
        beg, end = ragged_index[v:v + 2]
        feat_ss[beg:end, m] -= data_row[beg:end]
        meas_ss[v, m] -= data_row[beg:end].sum()


class TreeCatTrainer(object):
    """Class for training a TreeCat model."""

    def __init__(self, ragged_index, data, config):
        """Initialize a model in an unassigned state.

        Args:
          ragged_index: A [V+1]-shaped numpy array of indices into the ragged
            data array.
          data: An [N, _]-shaped numpy array of ragged data, where the vth
            column is stored in data[:, ragged_index[v]:ragged_index[v+1]].
          config: A global config dict.
        """
        logger.info('TreeCatTrainer of %d x %d data', data[0].shape[0],
                    len(data))
        ragged_index = np.asarray(ragged_index, np.int32)
        data = np.asarray(data, np.int8)
        config = config.copy()
        V = len(ragged_index) - 1  # Number of features, i.e. vertices.
        N = data.shape[0]  # Number of rows.
        assert V <= 32768, 'Invalid # features > 32768: {}'.format(V)
        assert len(data.shape) == 2
        assert data.shape[1] == ragged_index[-1]
        self._data = data
        self._config = config
        self._ragged_index = ragged_index
        self._assigned_rows = set()
        self._assignments = np.zeros([N, V], dtype=np.int8)
        self._tree = TreeStructure(V)
        assert self._tree.num_vertices == V
        self._program = make_propagation_program(self._tree.tree_grid)

        # These are useful dimensions to import into locals().
        E = V - 1  # Number of edges in the tree.
        K = V * (V - 1) // 2  # Number of edges in the complete graph.
        M = self._config['model_num_clusters']  # Clusters per latent.
        assert M <= 128, 'Invalid model_num_clusters > 128: {}'.format(M)
        self._VEKM = (V, E, K, M)

        # Use Jeffreys priors.
        self._vert_prior = 0.5
        self._edge_prior = 0.5 / M
        self._feat_prior = 0.5 / M
        self._meas_prior = self._feat_prior * np.array(
            [(ragged_index[v + 1] - ragged_index[v]) for v in range(V)],
            dtype=np.float32).reshape((V, 1))

        # Sufficient statistics are maintained always.
        self._vert_ss = np.zeros([V, M], np.int32)
        self._edge_ss = np.zeros([E, M, M], np.int32)
        self._feat_ss = np.zeros([self._ragged_index[-1], M], np.int32)
        self._meas_ss = np.zeros([V, M], np.int32)

        # Temporaries.
        self._vert_probs = np.empty(self._vert_ss.shape, np.float32)
        self._edge_probs = np.empty(self._edge_ss.shape, np.float32)
        self._feat_probs = np.empty(self._feat_ss.shape, np.float32)
        self._meas_probs = np.empty(self._meas_ss.shape, np.float32)

    @profile
    def add_row(self, row_id):
        logger.debug('TreeCatTrainer.add_row %d', row_id)
        assert row_id not in self._assigned_rows, row_id
        self._assigned_rows.add(row_id)

        # This is a little silly: we copy the entire model.
        np.add(self._vert_ss, self._vert_prior, out=self._vert_probs)
        np.add(self._edge_ss, self._edge_prior, out=self._edge_probs)
        np.add(self._feat_ss, self._feat_prior, out=self._feat_probs)
        np.add(self._meas_ss, self._meas_prior, out=self._meas_probs)

        jit_add_row(
            self._ragged_index,
            self._data[row_id, :],
            self._tree.tree_grid,
            self._program,
            self._assignments[row_id, :],
            self._vert_ss,
            self._edge_ss,
            self._feat_ss,
            self._meas_ss,
            self._vert_probs,
            self._edge_probs,
            self._feat_probs,
            self._meas_probs, )

    @profile
    def remove_row(self, row_id):
        logger.debug('TreeCatTrainer.remove_row %d', row_id)
        assert row_id in self._assigned_rows, row_id
        self._assigned_rows.remove(row_id)

        jit_remove_row(
            self._ragged_index,
            self._data[row_id, :],
            self._tree.tree_grid,
            self._assignments[row_id, :],
            self._vert_ss,
            self._edge_ss,
            self._feat_ss,
            self._meas_ss, )

    def get_edges(self):
        return [tuple(edge) for edge in self._tree.tree_grid[1:3, :].T]

    def set_edges(self, edges):
        V, E, K, M = self._VEKM
        self._tree.set_edges(edges)
        assignments = self._assignments[sorted(self._assigned_rows), :]
        for e, v1, v2 in self._tree.tree_grid.T:
            self._edge_ss[e, :, :] = count_pairs(assignments, v1, v2, M)
        self._program = make_propagation_program(self._tree.tree_grid)

    def get_edge_logits(self):
        """Compute non-normalized logprob of all V(V-1)/2 candidate edges.

        This is used for sampling and estimating the latent tree.
        """
        V, E, K, M = self._VEKM
        if len(self._assigned_rows) == V:
            assignments = self._assignments
        else:
            assignments = self._assignments[sorted(self._assigned_rows), :]
        vert_logits = logprob_dc(self._vert_ss, self._vert_prior, axis=1)
        edge_logits = np.zeros([K], np.float32)
        for k, v1, v2 in self._tree.complete_grid.T:
            counts = count_pairs(assignments, v1, v2, M)
            edge_logits[k] = (logprob_dc(counts, self._edge_prior) -
                              vert_logits[v1] - vert_logits[v2])
        return edge_logits

    @profile
    def sample_tree(self):
        """Samples a random tree.

        Returns:
          A pair (edges, edge_logits), where:
            edges: A list of (vertex, vertex) pairs.
            edge_logits: A [K]-shaped numpy array of edge logits.
        """
        logger.info('TreeCatTrainer.sample_tree given %d rows',
                    len(self._assigned_rows))
        complete_grid = self._tree.complete_grid
        edge_logits = self.get_edge_logits()
        assert edge_logits.shape[0] == complete_grid.shape[1]
        assert edge_logits.dtype == np.float32
        edges = self.get_edges()
        edges = sample_tree(
            complete_grid,
            edge_logits,
            edges,
            steps=self._config['learning_sample_tree_steps'])
        return edges, edge_logits

    def estimate_tree(self):
        """Compute a maximum likelihood tree.

        Returns:
          A pair (edges, edge_logits), where:
            edges: A list of (vertex, vertex) pairs.
            edge_logits: A [K]-shaped numpy array of edge logits.
        """
        logger.info('TreeCatTrainer.estimate_tree given %d rows',
                    len(self._assigned_rows))
        complete_grid = self._tree.complete_grid
        edge_logits = self.get_edge_logits()
        edges = estimate_tree(complete_grid, edge_logits)
        return edges, edge_logits

    def logprob(self):
        """Compute non-normalized log probability of data and assignments.

        This is used for testing goodness of fit of the category kernel.
        """
        assert len(self._assigned_rows) == self._assignments.shape[0]
        V, E, K, M = self._VEKM
        vertex_logits = logprob_dc(self._vert_ss, self._vert_prior, axis=1)
        logprob = vertex_logits.sum()
        for e, v1, v2 in self._tree.tree_grid.T:
            logprob += (logprob_dc(self._edge_ss[e, :, :], self._edge_prior) -
                        vertex_logits[v1] - vertex_logits[v2])
        for v in range(V):
            beg, end = self._ragged_index[v:v + 2]
            logprob += logprob_dc(self._feat_ss[beg:end, :], self._feat_prior)
            logprob -= logprob_dc(self._meas_ss[v, :], self._meas_prior[v])
        return logprob

    def train(self):
        """Train a TreeCat model using subsample-annealed MCMC.


        Returns:
          A trained model as a dictionary with keys:
            tree: A TreeStructure instance with the learned latent structure.
            suffstats: Sufficient statistics of features, vertices, and
              edges and a ragged_index for the features array.
            assignments: An [N, V]-shaped numpy array of latent cluster ids for
              each cell in the dataset, where N be the number of data rows and
              V is the number of features.
        """
        logger.info('TreeCatTrainer.train')
        set_random_seed(self._config['seed'])
        init_epochs = self._config['learning_init_epochs']
        full_epochs = self._config['learning_full_epochs']

        # Initialize using subsample annealing.
        num_rows = self._assignments.shape[0]
        for action, row_id in get_annealing_schedule(num_rows, init_epochs):
            if action == 'add_row':
                self.add_row(row_id)
            elif action == 'remove_row':
                self.remove_row(row_id)
            elif action == 'sample_tree':
                edges, edge_logits = self.sample_tree()
                self.set_edges(edges)
            else:
                raise ValueError(action)

        # Run full gibbs scans.
        for step in range(full_epochs):
            edges, edge_logits = self.sample_tree()
            self.set_edges(edges)
            for row_id in range(num_rows):
                self.remove_row(row_id)
                self.add_row(row_id)

        # Compute optimal tree.
        edges, edge_logits = self.estimate_tree()
        if self._config['learning_estimate_tree']:
            self.set_edges(edges)

        self._tree.gc()
        return {
            'config': self._config,
            'tree': self._tree,
            'edge_logits': edge_logits,
            'assignments': self._assignments,
            'suffstats': {
                'ragged_index': self._ragged_index,
                'vert_ss': self._vert_ss,
                'edge_ss': self._edge_ss,
                'feat_ss': self._feat_ss,
                'meas_ss': self._meas_ss,
            }
        }


def train_model(ragged_index, data, config):
    """Train a TreeCat model using subsample-annealed MCMC.

    Let N be the number of data rows and V be the number of features.

    Args:
      ragged_index: A [V+1]-shaped numpy array of indices into the ragged
        data array.
      data: An [N, _]-shaped numpy array of ragged data, where the vth
        column is stored in data[:, ragged_index[v]:ragged_index[v+1]].
      data: A list of numpy arrays, where each array is an N x _ column of
        counts of multinomial data.
      config: A global config dict.

    Returns:
      A trained model as a dictionary with keys:
        tree: A TreeStructure instance with the learned latent structure.
        suffstats: Sufficient statistics of features, vertices, and
          edges.
        assignments: An [N, V] numpy array of latent cluster ids for each
          cell in the dataset.
    """
    return TreeCatTrainer(ragged_index, data, config).train()


def _train_model(task):
    ragged_index, data, config = task
    return train_model(ragged_index, data, config)


def train_ensemble(ragged_index, data, config):
    """Train a TreeCat ensemble model using subsample-annealed MCMC.

    The ensemble size is controlled by config['model_ensemble_size'].
    Let N be the number of data rows and V be the number of features.

    Args:
      ragged_index: A [V+1]-shaped numpy array of indices into the ragged
        data array.
      data: An [N, _]-shaped numpy array of ragged data, where the vth
        column is stored in data[:, ragged_index[v]:ragged_index[v+1]].
      data: A list of numpy arrays, where each array is an N x _ column of
        counts of multinomial data.
      config: A global config dict.

    Returns:
      A trained model as a dictionary with keys:
        tree: A TreeStructure instance with the learned latent structure.
        suffstats: Sufficient statistics of features, vertices, and
          edges.
        assignments: An [N, V] numpy array of latent cluster ids for each
          cell in the dataset.
    """
    tasks = []
    for sub_seed in range(config['model_ensemble_size']):
        sub_config = config.copy()
        sub_config['seed'] += sub_seed
        tasks.append((ragged_index, data, sub_config))
    return parallel_map(_train_model, tasks)
