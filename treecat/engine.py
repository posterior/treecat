from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from six.moves import intern
from treecat.structure import make_propagation_schedule, make_tree
from treecat.util import TODO

DEFAULT_CONFIG = {
    'num_components': 32,
    'num_categories': 3,  # E.g. CSE-IT data.
    'seed': 0,
}

_ACTION_ADD = intern('ACTION_ADD')
_ACTION_REMOVE = intern('ACTION_REMOVE')
_ACTION_STRUCTURE = intern('ACTION_STRUCTURE')


class FeatureTree(object):
    def __init__(self, num_vertices, config):
        init_edges = [(v, v + 1) for v in range(num_vertices - 1)]
        V, E, grid = make_tree(init_edges)
        self._num_vertices = V
        self._num_edges = E
        self.update_grid(grid)

    def update_grid(self, grid):
        assert grid.shape == (3, self._num_edges)
        self._grid = grid
        self._tree_edges = {}
        self._tree_to_complete = np.zeros([self.num_edges], dtype=np.int32)
        for e, v1, v2 in self._grid.T:
            self._tree_edges[v1, v2] = e
            self._tree_edges[v2, v1] = e
            self._tree_to_complete[e] = v1 + v2 * (v2 + 1) // 2

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def grid(self):
        '''Array of (edge, vertex, vertex) triples defining the tree grahp.'''
        return self._grid

    @property
    def tree_to_complete(self):
        '''Index from tree edge ids e to complete edge ids k.'''
        return self._tree_to_complete

    @property
    def find_edge(self, v1, v2):
        return self._tree_edges[v1, v2]


def build_graph(tree, variables, config):
    '''Builds a tf graph for sampling assignments via message passing.

    Component distributions are Dirichlet-categorical.

    Names:
      row_data: Tensor of categorical values.
      row_mask: Tensor of presence/absence values for data.
      assignments: Latent mixture class assignments for a single row.
      update/add_row: Target op for adding a row of data.
      learn/edge_likelihood: Likelihoods of edges, used to learn structure.

    Returns:
      A dictionary of actions whose values can be input to Session.run().
    '''
    V = tree.num_vertices
    E = V - 1  # Number of edges in the tree.
    K = V * (V - 1) // 2  # Number of edges in the complete graph.
    M = config['num_components']
    C = config['num_categories']

    tree_to_complete = tf.constant(tree.tree_to_complete)
    assert tree_to_complete.shape == [E]

    row_data = tf.placeholder(dtype=tf.int32, shape=[V], name='row_data')
    row_mask = tf.placeholder(dtype=tf.int32, shape=[V], name='row_mask')
    prior = tf.constant(0.5, dtype=tf.float32, name='prior')

    if not variables:
        variables = {
            'vert_ss': tf.zeros([V, M], tf.int32),
            'edge_ss': tf.zeros([K, M, M], tf.int32),
            'feat_ss': tf.zeros([V, C, M], tf.int32),
        }

    # Sufficient statistics are maintained over the larger complete graph.
    vert_ss = tf.Variable(variables['vert_ss'], name='vert_ss')
    edge_ss = tf.Variable(variables['edge_ss'], name='edge_ss')
    feat_ss = tf.Variable(variables['feat_ss'], name='feat_ss')

    # Non-normalized probabilities are maintained over smaller the tree.
    vert_probs = tf.Variable(
        prior + tf.cast(vert_ss.initial_value, tf.float32), name='vert_probs')
    edge_probs = tf.Variable(
        prior + tf.cast(
            tf.gather(tree_to_complete, edge_ss.initial_value),
            tf.float32),
        name='edge_probs')

    actions = {
        'load': tf.global_variables_initializer(),
        'save': {
            'vert_ss': vert_ss,
            'edge_ss': edge_ss,
            'feat_ss': feat_ss
        },
    }

    with tf.name_scope('learn'):
        one = tf.constant(1.0, dtype=tf.float32, name='one')
        weights = tf.cast(edge_ss, tf.float32)
        logits = tf.lgamma(weights + prior) - tf.lgamma(weights + one)
        tf.reduce_sum(logits, [1, 2], name='edge_likelihood')

    with tf.name_scope('propagate'):
        schedule = make_propagation_schedule(tree.grid)
        messages = [None] * V
        samples = [None] * V
        with tf.name_scope('feature'):
            counts = tf.gather_nd(feat_ss, tf.stack([tf.range(V), row_data]))
            likelihood = prior + tf.cast(counts, tf.float32)
        with tf.name_scope('inbound'):
            for v, parent, children in reversed(schedule):
                prior_v = vert_probs[v, :]
                message = tf.cond(row_mask[v], likelihood[v] * prior_v,
                                  prior_v)
                for child in children:
                    e = tree.find_edge(v, child)
                    mat = edge_probs[e, :, :]
                    vec = messages[child, :, tf.newaxis]
                    message *= tf.reduce_sum(mat * vec) / prior_v
                messages[v] = message / tf.reduce_max(message)
        with tf.name_scope('outbound'):
            for v, parent, children in schedule:
                e = tree.find_edge(v, child)
                message = messages[v]
                if parent is not None:
                    prior_v = vert_probs[v, :]
                    mat = tf.transpose(edge_probs[e, :, :], [1, 0])
                    message *= tf.gather(samples[parent], mat) / prior_v
                sample = tf.squeeze(tf.multinomial(tf.log(message), 1), 1)
                samples[v] = sample
    assignments = tf.parallel_stack(samples, name='assignments')

    with tf.name_scope('update'):
        grid = tf.constant(tree.grid)
        vert_indices = tf.stack([assignments, row_data])
        edge_indices = tf.stack([
            grid[0, :, 0],
            tf.gather(assignments, grid[1, :]),
            tf.gather(assignments, grid[2, :]),
        ])
        feat_indices = tf.stack([tf.range(V), row_data, assignments])
        vert_ss = tf.scatter_add(vert_ss, vert_indices, row_mask, True)
        edge_ss = tf.scatter_add(edge_ss, edge_indices, row_mask, True)
        feat_ss = tf.scatter_add(feat_ss, feat_indices, row_mask, True)
        block = prior + tf.cast(
            tf.gather_nd(vert_ss, vert_indices), dtype=tf.float32)
        vert_probs = tf.scatter_update(vert_probs, vert_indices, block, True)
        block = prior + tf.cast(
            tf.gather_nd(edge_ss, edge_indices), dtype=tf.float32)
        edge_probs = tf.scatter_update(edge_probs, edge_indices, block, True)
        tf.group(
            vert_ss, edge_ss, feat_ss, vert_probs, edge_probs, name='add_row')

    return actions


class Model(object):
    def __init__(self, data, mask, config=None):
        assert len(data.shape) == 2
        assert data.shape == mask.shape
        if config is None:
            config = DEFAULT_CONFIG
        num_rows, num_features = data.shape
        self._config = config
        self._data = data
        self._mask = mask
        self._structure = FeatureTree(num_features, config)
        self._assignments = {}  # This maps id -> numpy array.
        self._seed = config['seed']
        self._variables = {}
        self._session = None
        self.update_session()

    def update_session(self):
        if self._session is not None:
            self._variables = self._session.run(self._actions['save'])
            self._session.close()
        with tf.Graph().as_default():
            tf.set_random_seed(self._seed)
            self._seed += 1
            self._actions = build_graph(self._structure, self._variables,
                                        self._config)
            self._session = tf.Session()
        self._session.run(self._actions['load'])

    def add_row(self, row_id):
        assert row_id not in self._assignments
        assignments, _ = self._session.run(
            ['assignments', 'update/add_row'],
            feed_dict={
                'row_data': self.data[row_id],
                'row_mask': self.mask[row_id],
            })
        self._assignments[row_id] = assignments

    def remove_row(self, row_id):
        assert row_id in self._assignments
        self._session.run(
            'update/add_row',
            feed_dict={
                'assignments': self._assignments[row_id],
                'row_data': self._data[row_id],
                'row_mask': -self._data[row_id],
            })

    def sample_structure(self):
        TODO()
        self.update_session()

    def sample(self):
        '''Sample the entire model using subsample annealed Gibbs sampling.'''
        self._assignments = {}  # Reset assignments.
        self._session.run('initialize')
        num_rows = self._dataframe.shape[0]
        for action, arg in get_annealing_schedule(num_rows, self._config):
            if action is _ACTION_ADD:
                self.add_row(arg)
            elif action is _ACTION_REMOVE:
                self.remove_row(arg)
            else:
                self.sample_structure()


def get_annealing_schedule(num_rows, config):
    TODO()
