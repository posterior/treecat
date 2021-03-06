from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from scipy.special import gammaln
from six import add_metaclass

from six.moves import range
from treecat.structure import OP_IN
from treecat.structure import OP_OUT
from treecat.structure import OP_ROOT
from treecat.structure import OP_UP
from treecat.structure import TreeStructure
from treecat.structure import estimate_tree
from treecat.structure import make_propagation_program
from treecat.structure import sample_tree
from treecat.tables import TY_MULTINOMIAL
from treecat.tables import Table
from treecat.util import SERIES
from treecat.util import TODO
from treecat.util import jit
from treecat.util import parallel_map
from treecat.util import prange
from treecat.util import profile
from treecat.util import sample_from_probs
from treecat.util import set_random_seed

logger = logging.getLogger(__name__)


def count_pairs(assignments, v1, v2, M):
    """Construct sufficient statistics for (v1, v2) pairs.

    Args:
        assignments: An _ x V assignment matrix with values in range(M).
        v1, v2: Column ids of the assignments matrix.
        M: The number of possible assignment bins.

    Returns:
        An M x M array of counts.
    """
    assert v1 != v2
    pairs = assignments[:, v1].astype(np.int32) * M + assignments[:, v2]
    return np.bincount(pairs, minlength=M * M).reshape((M, M))


def logprob_dc(counts, prior, axis=None):
    """Non-normalized log probability of a Dirichlet-Categorical distribution.

    See https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    """
    # Note that this excludes the factorial(counts) term, since we explicitly
    # track permutations in assignments.
    return gammaln(np.add(counts, prior, dtype=np.float32)).sum(axis)


def make_annealing_schedule(num_rows, epochs, sample_tree_rate):
    """Iterator for subsample annealing, yielding (action, arg) pairs.

    This generates a subsample annealing schedule starting from an empty
    assignment state (no rows are assigned). It then interleaves 'add_row' and
    'remove_row' actions so as to gradually increase the number of assigned
    rows. The increase rate is linear.

    Args:
        num_rows (int): Number of rows in dataset.
            The annealing schedule terminates when all rows are assigned.
        epochs (float): Number of epochs in the schedule (i.e. the number of
            times each datapoint is assigned). The fastest schedule is
            `epochs=1` which simply sequentially assigns all datapoints. More
            epochs takes more time.
        sample_tree_rate (float): The rate at which 'sample_tree' actions are
            generated. At `sample_tree_rate=1`, trees are sampled after each
            complete flushing of the subsample.

    Yields: (action, arg) pairs.
        Actions are one of: 'add_row', 'remove_row', or 'sample_tree'.
        When `action` is 'add_row' or 'remove_row', `arg` is a row_id in
        `range(num_rows)`. When `action` is 'sample_tree' arg is undefined.
    """
    assert epochs >= 1.0
    assert sample_tree_rate >= 1.0
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

    # Sample the tree sample_tree_rate times per batch.
    num_assigned = 0
    next_batch = 0
    while num_assigned < num_rows:
        if state >= 0.0:
            yield 'add_row', next(row_to_add)
            state -= remove_rate
            num_assigned += 1
            next_batch -= sample_tree_rate
        else:
            yield 'remove_row', next(row_to_remove)
            state += add_rate
            num_assigned -= 1
        if num_assigned > 0 and next_batch <= 0:
            yield 'sample_tree', None
            next_batch = num_assigned


@add_metaclass(ABCMeta)
class TreeTrainer(object):
    """Abstract base class for training a tree model various latent state.

    Derived classes must implement:
    - `add_row(row_id)`
    - `remove_row(row_id)`
    - `compute_edge_logits()`
    - `logprob()`
    """

    def __init__(self, N, V, tree_prior, config):
        """Initialize a model with an empty subsample.

        Args:
            N (int): Number of rows in the dataset.
            V (int): Number of columns (features) in the dataset.
            tree_prior: A [K]-shaped numpy array of prior edge log odds, where
                K is the number of edges in the complete graph on V vertices.
            config: A global config dict.
        """
        assert isinstance(N, int)
        assert isinstance(V, int)
        assert isinstance(tree_prior, np.ndarray)
        assert isinstance(config, dict)
        K = V * (V - 1) // 2  # Number of edges in complete graph.
        assert V <= 32768, 'Invalid # features > 32768: {}'.format(V)
        assert tree_prior.shape == (K, )
        assert tree_prior.dtype == np.float32
        self._config = config.copy()
        self._num_rows = N
        self._tree_prior = tree_prior
        self._tree = TreeStructure(V)
        assert self._tree.num_vertices == V
        self._program = make_propagation_program(self._tree.tree_grid)
        self._added_rows = set()

    @abstractmethod
    def add_row(self, row_id):
        """Add a given row to the current subsample."""

    @abstractmethod
    def remove_row(self, row_id):
        """Remove a given row from the current subsample."""

    @abstractmethod
    def compute_edge_logits(self):
        """Compute edge log probabilities on the complete graph."""

    @abstractmethod
    def logprob(self):
        """Compute non-normalized log probability of data and latent state.

        This is used for testing goodness of fit of the latent state kernel.
        This should only be called after training, i.e. after all rows have
        been added.
        """
        assert len(self._added_rows) == self._num_rows

    def get_edges(self):
        """Get a list of the edges in the current tree.

        Returns:
            An E-long list of (vertex,vertex) pairs.
        """
        return [tuple(edge) for edge in self._tree.tree_grid[1:3, :].T]

    def set_edges(self, edges):
        """Set edges of the latent structure and update statistics.

        Args:
            edges: An E-long list of (vertex,vertex) pairs.
        """
        self._tree.set_edges(edges)
        self._program = make_propagation_program(self._tree.tree_grid)

    @profile
    def sample_tree(self):
        """Samples a random tree.

        Returns:
            A pair (edges, edge_logits), where:
                edges: A list of (vertex, vertex) pairs.
                edge_logits: A [K]-shaped numpy array of edge logits.
        """
        logger.info('TreeCatTrainer.sample_tree given %d rows',
                    len(self._added_rows))
        SERIES.sample_tree_num_rows.append(len(self._added_rows))
        complete_grid = self._tree.complete_grid
        edge_logits = self.compute_edge_logits()
        assert edge_logits.shape[0] == complete_grid.shape[1]
        assert edge_logits.dtype == np.float32
        edges = self.get_edges()
        edges = sample_tree(complete_grid, edge_logits, edges)
        return edges, edge_logits

    def estimate_tree(self):
        """Compute a maximum likelihood tree.

        Returns:
            A pair (edges, edge_logits), where:
                edges: A list of (vertex, vertex) pairs.
                edge_logits: A [K]-shaped numpy array of edge logits.
        """
        logger.info('TreeCatTrainer.estimate_tree given %d rows',
                    len(self._added_rows))
        complete_grid = self._tree.complete_grid
        edge_logits = self.compute_edge_logits()
        edges = estimate_tree(complete_grid, edge_logits)
        return edges, edge_logits

    @profile
    def train(self):
        """Train a model using subsample-annealed MCMC.

        Returns:
            A trained model as a dictionary with keys:
                config: A global config dict.
                tree: A TreeStructure instance with the learned latent
                    structure.
                edge_logits: A [K]-shaped array of all edge logits.
        """
        logger.info('TreeTrainer.train')
        set_random_seed(self._config['seed'])
        init_epochs = self._config['learning_init_epochs']
        full_epochs = self._config['learning_full_epochs']
        sample_tree_rate = self._config['learning_sample_tree_rate']
        num_rows = self._num_rows

        # Initialize using subsample annealing.
        assert len(self._added_rows) == 0
        schedule = make_annealing_schedule(num_rows, init_epochs,
                                           sample_tree_rate)
        for action, row_id in schedule:
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
        assert len(self._added_rows) == num_rows
        for step in range(full_epochs):
            edges, edge_logits = self.sample_tree()
            self.set_edges(edges)
            for row_id in range(num_rows):
                self.remove_row(row_id)
                self.add_row(row_id)

        # Compute optimal tree.
        assert len(self._added_rows) == num_rows
        edges, edge_logits = self.estimate_tree()
        if self._config['learning_estimate_tree']:
            self.set_edges(edges)

        self._tree.gc()

        return {
            'config': self._config,
            'tree': self._tree,
            'edge_logits': edge_logits,
        }


@jit(nopython=True, cache=True)
def treecat_add_cell(
        feature_type,
        ragged_index,
        data_row,
        message,
        feat_probs,
        meas_probs,
        v, ):
    if feature_type == TY_MULTINOMIAL:
        beg, end = ragged_index[v:v + 2]
        feat_block = feat_probs[beg:end, :]
        meas_block = meas_probs[v, :]
        for c, count in enumerate(data_row[beg:end]):
            for _ in range(count):
                message *= feat_block[c, :]
                message /= meas_block
                feat_block[c, :] += np.float32(1)
                meas_block += np.float32(1)
    else:
        raise NotImplementedError


@profile
@jit(nopython=True, cache=True)
def treecat_add_row(
        feature_types,
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
            treecat_add_cell(
                feature_types[v],
                ragged_index,
                data_row,
                message,
                feat_probs,
                meas_probs,
                v, )
        elif op == OP_IN:
            # Propagate latent state inward from children to v.
            trans = edge_probs[e, :, :]
            if v > v2:
                trans = trans.T
            message *= np.dot(trans, messages[v2, :] / vert_probs[v2, :])
            message /= vert_probs[v, :]
            message /= message.max()  # Scale for numerical stability.
        elif op == OP_ROOT:
            # Process root node.
            assignments[v] = sample_from_probs(message)
        elif op == OP_OUT:
            # Propagate latent state outward from parent to v.
            trans = edge_probs[e, :, :]
            if v2 > v:
                trans = trans.T
            message *= trans[assignments[v2], :]
            message /= vert_probs[v, :]
            assignments[v] = sample_from_probs(message)

    # Update sufficient statistics.
    for v, m in enumerate(assignments):
        vert_ss[v, m] += 1
    for e in range(tree_grid.shape[1]):
        m1 = assignments[tree_grid[1, e]]
        m2 = assignments[tree_grid[2, e]]
        edge_ss[e, m1, m2] += 1
        edge_probs[e, m1, m2] += 1
    for v, m in enumerate(assignments):
        feature_type = feature_types[v]
        if feature_type == TY_MULTINOMIAL:
            beg, end = ragged_index[v:v + 2]
            feat_ss[beg:end, m] += data_row[beg:end]
            meas_ss[v, m] += data_row[beg:end].sum()
        else:
            raise NotImplementedError


@profile
@jit(nopython=True, cache=True)
def treecat_remove_row(
        feature_types,
        ragged_index,
        data_row,
        tree_grid,
        assignments,
        vert_ss,
        edge_ss,
        feat_ss,
        meas_ss,
        edge_probs, ):
    # Update sufficient statistics.
    for v, m in enumerate(assignments):
        vert_ss[v, m] -= 1
    for e in range(tree_grid.shape[1]):
        m1 = assignments[tree_grid[1, e]]
        m2 = assignments[tree_grid[2, e]]
        edge_ss[e, m1, m2] -= 1
        edge_probs[e, m1, m2] -= 1
    for v, m in enumerate(assignments):
        feature_type = feature_types[v]
        if feature_type == TY_MULTINOMIAL:
            beg, end = ragged_index[v:v + 2]
            feat_ss[beg:end, m] -= data_row[beg:end]
            meas_ss[v, m] -= data_row[beg:end].sum()
        else:
            raise NotImplementedError


@jit(nopython=True, cache=True)
def treecat_compute_edge_logit(M, gammaln_table, assign1, assign2):
    counts = np.zeros((M, M), np.int32)
    for n in range(assign1.shape[0]):
        counts[assign1[n], assign2[n]] += 1
    result = np.float32(0)
    for m1 in range(M):
        for m2 in range(M):
            result += gammaln_table[counts[m1, m2]]
    return result


@jit(nopython=True, parallel=True)
def treecat_compute_edge_logits_par(M, grid, gammaln_table, assignments,
                                    vert_logits):
    K = grid.shape[1]
    N, V = assignments.shape
    edge_logits = np.zeros(K, np.float32)
    for k in prange(K):
        v1, v2 = grid[1:3, k]
        assign1 = assignments[:, v1]
        assign2 = assignments[:, v2]
        edge_logit = treecat_compute_edge_logit(M, gammaln_table, assign1,
                                                assign2)
        edge_logits[k] = edge_logit - vert_logits[v1] - vert_logits[v2]
    return edge_logits


@jit(nopython=True, cache=True)
def treecat_compute_edge_logits_seq(M, grid, gammaln_table, assignments,
                                    vert_logits):
    K = grid.shape[1]
    N, V = assignments.shape
    edge_logits = np.zeros(K, np.float32)
    for k in range(K):
        v1, v2 = grid[1:3, k]
        assign1 = assignments[:, v1]
        assign2 = assignments[:, v2]
        edge_logit = treecat_compute_edge_logit(M, gammaln_table, assign1,
                                                assign2)
        edge_logits[k] = edge_logit - vert_logits[v1] - vert_logits[v2]
    return edge_logits


@profile
def treecat_compute_edge_logits(M, grid, gammaln_table, assignments,
                                vert_logits, parallel):
    if parallel:
        return treecat_compute_edge_logits_par(M, grid, gammaln_table,
                                               assignments, vert_logits)
    else:
        return treecat_compute_edge_logits_seq(M, grid, gammaln_table,
                                               assignments, vert_logits)


class TreeCatTrainer(TreeTrainer):
    """Class for training a TreeCat model."""

    def __init__(self, table, tree_prior, config):
        """Initialize a model with an empty subsample.

        Args:
            table: A Table instance holding N rows of V features of data.
            tree_prior: A [K]-shaped numpy array of prior edge log odds, where
                K is the number of edges in the complete graph on V vertices.
            config: A global config dict.
        """
        logger.info('TreeCatTrainer of %d x %d data', table.num_rows,
                    table.num_cols)
        assert isinstance(table, Table)
        N = table.num_rows  # Number of rows.
        V = table.num_cols  # Number of features, i.e. vertices.
        TreeTrainer.__init__(self, N, V, tree_prior, config)
        assert self._num_rows == N
        assert len(self._added_rows) == 0
        self._table = table
        self._assignments = np.zeros([N, V], dtype=np.int8)

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
            [(table.ragged_index[v + 1] - table.ragged_index[v])
             for v in range(V)],
            dtype=np.float32).reshape((V, 1))
        self._gammaln_table = gammaln(
            np.arange(1 + N, dtype=np.float32) + self._edge_prior)
        assert self._gammaln_table.dtype == np.float32

        # Sufficient statistics are maintained always.
        self._vert_ss = np.zeros([V, M], np.int32)
        self._edge_ss = np.zeros([E, M, M], np.int32)
        self._feat_ss = np.zeros([table.ragged_index[-1], M], np.int32)
        self._meas_ss = np.zeros([V, M], np.int32)

        # Temporaries.
        self._vert_probs = np.empty(self._vert_ss.shape, np.float32)
        self._edge_probs = np.empty(self._edge_ss.shape, np.float32)
        self._feat_probs = np.empty(self._feat_ss.shape, np.float32)
        self._meas_probs = np.empty(self._meas_ss.shape, np.float32)

        # Maintain edge_probs.
        np.add(self._edge_ss, self._edge_prior, out=self._edge_probs)

    @profile
    def add_row(self, row_id):
        logger.debug('TreeCatTrainer.add_row %d', row_id)
        assert row_id not in self._added_rows, row_id
        self._added_rows.add(row_id)

        # These are used for scratch work, so we create them each step.
        np.add(self._vert_ss, self._vert_prior, out=self._vert_probs)
        np.add(self._feat_ss, self._feat_prior, out=self._feat_probs)
        np.add(self._meas_ss, self._meas_prior, out=self._meas_probs)

        treecat_add_row(
            self._table.feature_types,
            self._table.ragged_index,
            self._table.data[row_id, :],
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
        assert row_id in self._added_rows, row_id
        self._added_rows.remove(row_id)

        treecat_remove_row(
            self._table.feature_types,
            self._table.ragged_index,
            self._table.data[row_id, :],
            self._tree.tree_grid,
            self._assignments[row_id, :],
            self._vert_ss,
            self._edge_ss,
            self._feat_ss,
            self._meas_ss,
            self._edge_probs, )

    def set_edges(self, edges):
        TreeTrainer.set_edges(self, edges)
        V, E, K, M = self._VEKM
        assignments = self._assignments[sorted(self._added_rows), :]
        for e, v1, v2 in self._tree.tree_grid.T:
            self._edge_ss[e, :, :] = count_pairs(assignments, v1, v2, M)
        np.add(self._edge_ss, self._edge_prior, out=self._edge_probs)

    @profile
    def compute_edge_logits(self):
        """Compute non-normalized logprob of all V(V-1)/2 candidate edges.

        This is used for sampling and estimating the latent tree.
        """
        V, E, K, M = self._VEKM
        vert_logits = logprob_dc(self._vert_ss, self._vert_prior, axis=1)
        if len(self._added_rows) == V:
            assignments = self._assignments
        else:
            assignments = self._assignments[sorted(self._added_rows), :]
        assignments = np.array(assignments, order='F')
        parallel = self._config['learning_parallel']
        result = treecat_compute_edge_logits(M, self._tree.complete_grid,
                                             self._gammaln_table, assignments,
                                             vert_logits, parallel)
        result += self._tree_prior
        return result

    def logprob(self):
        """Compute non-normalized log probability of data and latent state.

        This is used for testing goodness of fit of the latent state kernel.
        """
        assert len(self._added_rows) == self._num_rows
        V, E, K, M = self._VEKM
        vertex_logits = logprob_dc(self._vert_ss, self._vert_prior, axis=1)
        logprob = vertex_logits.sum()
        for e, v1, v2 in self._tree.tree_grid.T:
            logprob += (logprob_dc(self._edge_ss[e, :, :], self._edge_prior) -
                        vertex_logits[v1] - vertex_logits[v2])
        for v in range(V):
            beg, end = self._table.ragged_index[v:v + 2]
            logprob += logprob_dc(self._feat_ss[beg:end, :], self._feat_prior)
            logprob -= logprob_dc(self._meas_ss[v, :], self._meas_prior[v])
        return logprob

    def train(self):
        """Train a TreeCat model using subsample-annealed MCMC.

        Returns:
            A trained model as a dictionary with keys:
                config: A global config dict.
                tree: A TreeStructure instance with the learned latent
                    structure.
                edge_logits: A [K]-shaped array of all edge logits.
                suffstats: Sufficient statistics of features, vertices, and
                    edges and a ragged_index for the features array.
                assignments: An [N, V]-shaped numpy array of latent cluster
                    ids for each cell in the dataset, where N be the number of
                    data rows and V is the number of features.
        """
        model = TreeTrainer.train(self)
        model['assignments'] = self._assignments
        model['suffstats'] = {
            'ragged_index': self._table.ragged_index,
            'vert_ss': self._vert_ss,
            'edge_ss': self._edge_ss,
            'feat_ss': self._feat_ss,
            'meas_ss': self._meas_ss,
        }
        return model


def treegauss_add_row(
        data_row,
        tree_grid,
        program,
        latent_row,
        vert_ss,
        edge_ss,
        feat_ss, ):
    # Sample latent state using dynamic programming.
    TODO('https://github.com/posterior/treecat/issues/26')

    # Update sufficient statistics.
    for v in range(latent_row.shape[0]):
        z = latent_row[v, :]
        vert_ss[v, :, :] += np.outer(z, z)
    for e in range(tree_grid.shape[1]):
        z1 = latent_row[tree_grid[1, e], :]
        z2 = latent_row[tree_grid[2, e], :]
        edge_ss[e, :, :] += np.outer(z1, z2)
    for v, x in enumerate(data_row):
        if np.isnan(x):
            continue
        z = latent_row[v, :]
        feat_ss[v] += 1
        feat_ss[v, 1] += x
        feat_ss[v, 2:] += x * z  # TODO Use central covariance.


def treegauss_remove_row(
        data_row,
        tree_grid,
        latent_row,
        vert_ss,
        edge_ss,
        feat_ss, ):
    # Update sufficient statistics.
    for v in range(latent_row.shape[0]):
        z = latent_row[v, :]
        vert_ss[v, :, :] -= np.outer(z, z)
    for e in range(tree_grid.shape[1]):
        z1 = latent_row[tree_grid[1, e], :]
        z2 = latent_row[tree_grid[2, e], :]
        edge_ss[e, :, :] -= np.outer(z1, z2)
    for v, x in enumerate(data_row):
        if np.isnan(x):
            continue
        z = latent_row[v, :]
        feat_ss[v] -= 1
        feat_ss[v, 1] -= x
        feat_ss[v, 2:] -= x * z  # TODO Use central covariance.


class TreeGaussTrainer(TreeTrainer):
    """Class for training a TreeGauss model."""

    def __init__(self, data, tree_prior, config):
        """Initialize a model with an empty subsample.

        Args:
            data: An [N, V]-shaped numpy array of real-valued data.
            tree_prior: A [K]-shaped numpy array of prior edge log odds, where
                K is the number of edges in the complete graph on V vertices.
            config: A global config dict.
        """
        assert isinstance(data, np.ndarray)
        data = np.asarray(data, np.float32)
        assert len(data.shape) == 2
        N, V = data.shape
        D = config['model_latent_dim']
        E = V - 1  # Number of edges in the tree.
        TreeTrainer.__init__(self, N, V, tree_prior, config)
        self._data = data
        self._latent = np.zeros([N, V, D], np.float32)

        # This is symmetric positive definite.
        self._vert_ss = np.zeros([V, D, D], np.float32)
        # This is arbitrary (not necessarily symmetric).
        self._edge_ss = np.zeros([E, D, D], np.float32)
        # This represents (count, mean, covariance).
        self._feat_ss = np.zeros([V, D, 1 + 1 + D], np.float32)

    def add_row(self, row_id):
        logger.debug('TreeGaussTrainer.add_row %d', row_id)
        assert row_id not in self._added_rows, row_id
        self._added_rows.add(row_id)

        treegauss_add_row(
            self._data[row_id, :],
            self._tree.tree_grid,
            self._program,
            self._latent[row_id, :, :],
            self._vert_ss,
            self._edge_ss,
            self._feat_ss, )

    def remove_row(self, row_id):
        logger.debug('TreeGaussTrainer.remove_row %d', row_id)
        assert row_id in self._added_rows, row_id
        self._added_rows.remove(row_id)

        treecat_remove_row(
            self._data[row_id, :],
            self._tree.tree_grid,
            self._latent[row_id, :, :],
            self._vert_ss,
            self._edge_ss,
            self._feat_s, )

    def set_edges(self, edges):
        TreeTrainer.set_edges(self, edges)
        latent = self._latent[sorted(self._added_rows), :, :]
        for e, v1, v2 in self._tree.tree_grid.T:
            self._edge_ss[e, :, :] = np.dot(latent[:, v1, :].T,
                                            latent[:, v2, :])

    def compute_edge_logits(self):
        """Compute non-normalized logprob of all V(V-1)/2 candidate edges.

        This is used for sampling and estimating the latent tree.
        """
        TODO('https://github.com/posterior/treecat/issues/26')

    def logprob(self):
        """Compute non-normalized log probability of data and latent state.

        This is used for testing goodness of fit of the latent state kernel.
        """
        assert len(self._added_rows) == self._num_rows
        TODO('https://github.com/posterior/treecat/issues/26')

    def train(self):
        """Train a TreeGauss model using subsample-annealed MCMC.

        Returns:
            A trained model as a dictionary with keys:
                config: A global config dict.
                tree: A TreeStructure instance with the learned latent
                    structure.
                edge_logits: A [K]-shaped array of all edge logits.
                suffstats: Sufficient statistics of features and vertices.
                latent: An [N, V, M]-shaped numpy array of latent states, where
                    N is the number of data rows, V is the number of features,
                    and M is the dimension of each latent variable.
        """
        model = TreeTrainer.train(self)
        model['latent'] = self._latent
        model['suffstats'] = {
            'vert_ss': self._vert_ss,
            'edge_ss': self._edge_ss,
            'feat_ss': self._feat_ss,
        }
        return model


class TreeMogTrainer(TreeTrainer):
    """Class for training a tree mixture-of-Gaussians model."""

    def __init__(self, data, tree_prior, config):
        TODO('https://github.com/posterior/treecat/issues/27')

    def add_row(self, row_id):
        """Add a given row to the current subsample."""
        TODO('https://github.com/posterior/treecat/issues/27')

    def remove_row(self, row_id):
        """Remove a given row from the current subsample."""
        TODO('https://github.com/posterior/treecat/issues/27')

    def compute_edge_logits(self):
        """Compute edge log probabilities on the complete graph."""
        TODO('https://github.com/posterior/treecat/issues/27')

    def logprob(self):
        """Compute non-normalized log probability of data and latent state."""
        assert len(self._added_rows) == self._num_rows
        TODO('https://github.com/posterior/treecat/issues/27')


def train_model(table, tree_prior, config):
    """Train a TreeCat model using subsample-annealed MCMC.

    Let N be the number of data rows and V be the number of features.

    Args:
        table: A Table instance holding N rows of V features of data.
        tree_prior: A [K]-shaped numpy array of prior edge log odds, where
            K is the number of edges in the complete graph on V vertices.
        config: A global config dict.

    Returns:
        A trained model as a dictionary with keys:
            config: A global config dict.
            tree: A TreeStructure instance with the learned latent structure.
            edge_logits: A [K]-shaped array of all edge logits.
            suffstats: Sufficient statistics of features, vertices, and edges.
            assignments: An [N, V] numpy array of latent cluster ids for each
                cell in the dataset.
    """
    assert isinstance(table, Table)
    M = config['model_num_clusters']
    D = config['model_latent_dim']
    assert M >= 1
    assert D >= 0
    if D == 0:
        Trainer = TreeCatTrainer
    elif M == 1:
        Trainer = TreeGaussTrainer
    else:
        Trainer = TreeMogTrainer
    return Trainer(table, tree_prior, config).train()


def _train_model(task):
    table, tree_prior, config = task
    return train_model(table, tree_prior, config)


def train_ensemble(table, tree_prior, config):
    """Train a TreeCat ensemble model using subsample-annealed MCMC.

    The ensemble size is controlled by config['model_ensemble_size'].
    Let N be the number of data rows and V be the number of features.

    Args:
        table: A Table instance holding N rows of V features of data.
        tree_prior: A [K]-shaped numpy array of prior edge log odds, where
            K is the number of edges in the complete graph on V vertices.
        config: A global config dict.

    Returns:
        A trained model as a dictionary with keys:
            tree: A TreeStructure instance with the learned latent structure.
            suffstats: Sufficient statistics of features, vertices, and edges.
            assignments: An [N, V] numpy array of latent cluster ids for each
                cell in the dataset.
    """
    tasks = []
    for sub_seed in range(config['model_ensemble_size']):
        sub_config = config.copy()
        sub_config['seed'] += sub_seed
        tasks.append((table, tree_prior, sub_config))
    return parallel_map(_train_model, tasks)
