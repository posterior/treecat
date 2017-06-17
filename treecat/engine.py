from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging

import numpy as np
import tensorflow as tf

from treecat.structure import make_complete_graph
from treecat.structure import make_propagation_schedule
from treecat.structure import make_tree
from treecat.structure import sample_tree
from treecat.util import profile_timed

DEFAULT_CONFIG = {
    'seed': 0,
    'num_categories': 3,  # E.g. CSE-IT data.
    'num_components': 32,
    'annealing': {
        'init_rows': 2,
        'epochs': 100.0,
    },
}

logger = logging.getLogger(__name__)

# line_profiler defines profile in the __builtins__ module.
profile = getattr(__builtins__, 'profile', lambda fun: fun)


class FeatureTree(object):
    '''Topological data representing a tree on features.'''

    def __init__(self, num_vertices):
        logger.debug('FeatureTree with %d vertices', num_vertices)
        init_edges = [(v, v + 1) for v in range(num_vertices - 1)]
        V, E, tree_grid = make_tree(init_edges)
        V, K, complete_grid = make_complete_graph(V)
        self._num_vertices = V
        self._num_edges = E
        self._complete_grid = complete_grid
        self.update_grid(tree_grid)

    def __eq__(self, other):
        return (self._num_vertices == other._num_vertices and
                (self._tree_grid == other._tree_grid).all())

    def update_grid(self, tree_grid):
        assert tree_grid.shape == (3, self._num_edges)
        self._tree_grid = tree_grid
        self._tree_edges = {}
        for e, v1, v2 in self._tree_grid.T:
            self._tree_edges[v1, v2] = e
            self._tree_edges[v2, v1] = e

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def tree_grid(self):
        '''Array of (edge, vertex, vertex) triples defining the tree graph.'''
        return self._tree_grid

    @property
    def complete_grid(self):
        '''Array of (edge,vertex,vertex) triples defining a complete graph.'''
        return self._complete_grid

    def find_edge(self, v1, v2):
        return self._tree_edges[v1, v2]


@profile_timed
def build_graph(tree, inits, config):
    '''Builds a tf graph for sampling assignments via message passing.

    Component distributions are Dirichlet-categorical.

    Args:
      tree: A FeatureTree object.
      inits: An dict of optional saved tensor values.
      config: A global config dict.

    Names:
      row_data: Tensor of categorical values.
      row_mask: Tensor of presence/absence values for data.
        Should be positive to add a row and negative to remove a row.
      assignments: Latent mixture class assignments for a single row.
      assign/add_row: Target op for adding a row of data.
      assign/remove_row: Target op for removing a row of data.
      structure/edge_prob: Non-normalized probabilities of edges.

    Returns:
      A dictionary of actions whose values can be input to Session.run():
        load: Initialize global variables.
        save: Eval global variables.
    '''
    logger.debug('build_graph of tree with %d vertices' % tree.num_vertices)
    assert isinstance(tree, FeatureTree)
    assert isinstance(inits, dict)
    assert isinstance(config, dict)
    V = tree.num_vertices
    E = V - 1  # Number of edges in the tree.
    K = V * (V - 1) // 2  # Number of edges in the complete graph.
    M = config['num_components']  # Components of the mixture model.
    C = config['num_categories']
    vertices = tf.range(V, dtype=tf.int32)
    tree_grid = tf.constant(tree.tree_grid)
    complete_grid = tf.constant(tree.complete_grid)

    # Hard-code these hyperparameters.
    feat_prior = tf.constant(0.5, dtype=tf.float32)  # Jeffreys prior.
    vert_prior = tf.constant(1.0 / M, dtype=tf.float32)  # Nonparametric.
    edge_prior = tf.constant(1.0 / M**2, dtype=tf.float32)  # Nonparametric.

    # Sufficient statistics are maintained always (across and within batches).
    vert_ss = tf.Variable(inits.get('vert_ss', tf.zeros([V, M], tf.int32)))
    edge_ss = tf.Variable(inits.get('edge_ss', tf.zeros([E, M, M], tf.int32)))
    feat_ss = tf.Variable(inits.get('feat_ss', tf.zeros([V, C, M], tf.int32)))

    # Sufficient statistics for tree learning are maintained within each batch.
    # This is the most expensive data structure, costing O(V^2 M^2) space.
    tree_ss = tf.Variable(tf.zeros([K, M, M], tf.int32), name='tree_ss')

    # Non-normalized probabilities are maintained within each batch.
    vert_probs = tf.Variable(
        vert_prior + tf.cast(vert_ss.initial_value, tf.float32),
        name='vert_probs')
    edge_probs = tf.Variable(
        edge_prior + tf.cast(edge_ss.initial_value, tf.float32),
        name='edge_probs')

    # This is run to compute edge logprobs before learning the tree structure.
    with tf.name_scope('structure'):
        one = tf.constant(1.0, dtype=tf.float32, name='one')
        weights = tf.cast(tree_ss, tf.float32)
        logits = tf.lgamma(weights + edge_prior) - tf.lgamma(weights + one)
        edge_logprob = tf.reduce_sum(logits, [1, 2])
        edge_logprob -= tf.reduce_max(edge_logprob)
        tf.exp(edge_logprob, name='edge_prob')

    # This is run only during add_row().
    row_data = tf.placeholder(dtype=tf.int32, shape=[V], name='row_data')
    row_mask = tf.placeholder(dtype=tf.bool, shape=[V], name='row_mask')
    with tf.name_scope('propagate'):
        schedule = make_propagation_schedule(tree.tree_grid)
        messages = [None] * V
        samples = [None] * V
        with tf.name_scope('feature'):
            indices = tf.stack([vertices, row_data], 1)
            counts = tf.gather_nd(feat_ss, indices)
            likelihood = feat_prior + tf.cast(counts, tf.float32)
        with tf.name_scope('inbound'):
            for v, parent, children in reversed(schedule):
                prior_v = vert_probs[v, :]
                message = tf.cond(row_mask[v], lambda: prior_v,
                                  lambda: likelihood[v] * prior_v)
                for child in children:
                    e = tree.find_edge(v, child)
                    mat = edge_probs[e, :, :]
                    vec = messages[child][:, tf.newaxis]
                    message *= tf.reduce_sum(mat * vec) / prior_v
                messages[v] = message / tf.reduce_max(message)
        with tf.name_scope('outbound'):
            for v, parent, children in schedule:
                message = messages[v]
                if parent is not None:
                    e = tree.find_edge(v, parent)
                    prior_v = vert_probs[v, :]
                    mat = tf.transpose(edge_probs[e, :, :], [1, 0])
                    message *= tf.gather(mat, samples[parent])[0, :] / prior_v
                    assert len(message.shape) == 1
                sample = tf.cast(
                    tf.squeeze(
                        tf.multinomial(tf.log(message)[tf.newaxis, :], 1), 1),
                    tf.int32)
                samples[v] = sample
    assignments = tf.squeeze(tf.parallel_stack(samples), name='assignments')

    # This is run during add_row() and remove_row().
    with tf.name_scope('update'):
        # This is optimized for the dense case, i.e. row_mask is mostly True.
        feat_indices = tf.stack([vertices, row_data, assignments], 1)
        vert_indices = tf.stack([vertices, assignments], 1)
        edge_indices = tf.stack([
            tree_grid[0, :],
            tf.gather(assignments, tree_grid[1, :]),
            tf.gather(assignments, tree_grid[2, :]),
        ], 1)
        tree_indices = tf.stack([
            complete_grid[0, :],
            tf.gather(assignments, complete_grid[1, :]),
            tf.gather(assignments, complete_grid[2, :]),
        ], 1)
        delta_v_int = tf.cast(row_mask, tf.int32)
        delta_v_float = tf.cast(row_mask, tf.float32)
        delta_e_int = tf.cast(
            tf.logical_and(
                tf.gather(row_mask, tree_grid[1, :]),
                tf.gather(row_mask, tree_grid[2, :])), tf.int32)
        delta_e_float = tf.cast(delta_e_int, tf.float32)
        delta_k_int = tf.cast(
            tf.logical_and(
                tf.gather(row_mask, complete_grid[1, :]),
                tf.gather(row_mask, complete_grid[2, :])), tf.int32)
        tf.group(
            tf.scatter_nd_add(feat_ss, feat_indices, delta_v_int, True),
            tf.scatter_nd_add(vert_ss, vert_indices, delta_v_int, True),
            tf.scatter_nd_add(edge_ss, edge_indices, delta_e_int, True),
            tf.scatter_nd_add(tree_ss, tree_indices, delta_k_int, True),
            tf.scatter_nd_add(vert_probs, vert_indices, delta_v_float, True),
            tf.scatter_nd_add(edge_probs, edge_indices, delta_e_float, True),
            name='add_row')
        tf.group(
            tf.scatter_nd_sub(feat_ss, feat_indices, delta_v_int, True),
            tf.scatter_nd_sub(vert_ss, vert_indices, delta_v_int, True),
            tf.scatter_nd_sub(edge_ss, edge_indices, delta_e_int, True),
            # Note that tree_ss is not updated during remove_row.
            tf.scatter_nd_sub(vert_probs, vert_indices, delta_v_float, True),
            tf.scatter_nd_sub(edge_probs, edge_indices, delta_e_float, True),
            name='remove_row')

    # These actions allow saving and loading variables when the graph changes.
    return {
        'load': tf.global_variables_initializer(),
        'save': {
            'feat_ss': feat_ss,
            'vert_ss': vert_ss,
            'edge_ss': edge_ss,
        },
    }


class Model(object):
    '''Main TreeCat model class.'''

    def __init__(self, data, mask, config=None):
        '''Initialize a model in an unassigned state.

        Args:
            data: A 2D array of categorical data.
            mask: A 2D array of presence/absence, where present = True.
            config: A global config dict.
        '''
        logger.info('Model of %d x %d data', len(data), len(data[0]))
        data = np.asarray(data, np.int32)
        mask = np.asarray(mask, np.bool_)
        num_rows, num_features = data.shape
        assert data.shape == mask.shape
        if config is None:
            config = DEFAULT_CONFIG
        self._data = data
        self._mask = mask
        self._config = config
        self._seed = config['seed']
        self._assignments = {}  # This maps id -> numpy array.
        self._variables = {}
        self._structure = FeatureTree(num_features)
        self._initialize()

    def _initialize(self):
        self._session = None
        self._update_session()

    @profile_timed
    def _update_session(self):
        if self._session is not None:
            self._variables = self._session.run(self._actions['save'])
            self._session.close()
        with tf.Graph().as_default():
            tf.set_random_seed(self._seed)
            self._actions = build_graph(self._structure, self._variables,
                                        self._config)
            self._session = tf.Session()
        self._session.run(self._actions['load'])

    @profile_timed
    def _add_row(self, row_id):
        logger.debug('Model.add_row %d', row_id)
        assert row_id not in self._assignments, row_id
        assignments, _ = self._session.run(
            ['assignments:0', 'update/add_row'],
            feed_dict={
                'row_data:0': self._data[row_id],
                'row_mask:0': self._mask[row_id],
            })
        assert assignments.shape == (self._data.shape[1], )
        self._assignments[row_id] = assignments

    @profile_timed
    def _remove_row(self, row_id):
        logger.debug('Model.remove_row %d', row_id)
        assert row_id in self._assignments, row_id
        self._session.run(
            'update/add_row',
            feed_dict={
                'assignments:0': self._assignments.pop(row_id),
                'row_data:0': self._data[row_id],
                'row_mask:0': self._mask[row_id],
            })

    @profile_timed
    def _sample_structure(self):
        logger.info('Model._sample_structure given %d rows',
                    len(self._assignments))
        edge_prob = self._session.run('structure/edge_prob:0')
        complete_grid = self._structure.complete_grid
        assert edge_prob.shape[0] == complete_grid.shape[1]
        edges = self._structure.tree_grid[1:3, :].T
        edges = sample_tree(complete_grid, edge_prob, edges, self._seed)
        self._seed += 1
        V, E, grid = make_tree(edges)
        self._structure.update_grid(grid)
        self._update_session()

    def fit(self):
        '''Sample the entire model using subsample-annealed MCMC.'''
        logger.info('Model.fit')
        assert not self._assignments, 'assignments have already been sampled'
        num_rows = self._data.shape[0]
        for action, row_id in get_annealing_schedule(num_rows, self._config):
            if action == 'add_row':
                self._add_row(row_id)
            elif action == 'remove_row':
                self._remove_row(row_id)
            else:
                self._sample_structure()

    _pickle_attrs = [
        '_data',
        '_mask',
        '_config',
        '_seed',
        '_assignments',
        '_variables',
        '_structure',
    ]

    def __getstate__(self):
        logger.info('Model.__getstate__')
        if self._session is not None:
            self._variables = self._session.run(self._actions['save'])
        return {key: getattr(self, key) for key in self._pickle_attrs}

    def __setstate__(self, state):
        logger.info('Model.__setstate__')
        self.__dict__.update(state)
        self._initialize()


def get_annealing_schedule(num_rows, config):
    '''Iterator for subsample annealing yielding (action, arg) pairs.'''
    # Randomly shuffle rows.
    row_ids = list(range(num_rows))
    np.random.seed(config['seed'])
    np.random.shuffle(row_ids)
    row_to_add = itertools.cycle(row_ids)
    row_to_remove = itertools.cycle(row_ids)

    # Use a linear annealing schedule.
    epochs = float(config['annealing']['epochs'])
    add_rate = epochs
    remove_rate = epochs - 1.0
    state = epochs * config['annealing']['init_rows']

    # Perform batch operations between batches.
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
            yield 'batch', None
            num_stale = num_fresh
            num_fresh = 0
