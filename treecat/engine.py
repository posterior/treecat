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
        self._grid = grid

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def grid(self):
        return self._grid

    def _sort_vertices(self):
        '''Root-first depth-first topological sort.'''
        # TODO This is more parallelizable when the root is central.
        V = self._num_vertices
        neighbors = [set() for _ in range(V)]
        for v1, v2 in self._edges:
            neighbors[v1].add(v2)
            neighbors[v2].add(v1)
        order = []
        pending = set([0])
        done = set()
        while pending:
            v = min(pending)
            order.append(v)
            pending.remove(v)
            done.add(v)
            pending |= neighbors[v] - done
        TODO('create self._schedule for propagation')

    def init_topology(self):
        # Topological structure is initialized arbitrarily.
        # TODO Add confusion matrices to edges.
        self._edges = [(i, i + 1) for i in range(self._num_vertices - 1)]
        self._neighbors = TODO()

    def sample_topology(self):
        TODO()


def build_graph(tree, config):
    '''Builds a tf graph for sampling assignments via message passing.

    Component distributions are Dirichlet-categorical.

    Names:
      row_data: Tensor of categorical values.
      row_mask: Tensor of presence/absence values for data.
      assignments: Latent mixture class assignments for a single row.
      update/add_row: Target op for adding a row of data.
      learn/edge_likelihood: Likelihoods of edges, used to learn structure.

    Returns:
      An op to initialize global variables.
    '''
    V = tree.num_vertices
    E = V - 1  # Number of edges in the tree.
    K = V * (V - 1) // 2  # Number of edges in the complete graph.
    M = config['num_components']
    C = config['num_categories']
    all_edges = np.zeros([V, V], dtype=np.int32)
    for k, v1, v2 in tree.complete_graph.T:
        all_edges[v1, v2] = k
        all_edges[v2, v1] = k
    active_edges = np.zeros([E], dtype=np.int32)
    for e, v1, v2 in tree.grid.T:
        active_edges[e] = all_edges[v1, v2]

    row_data = tf.placeholder(dtype=tf.int32, shape=[V], name='row_data')
    row_mask = tf.placeholder(dtype=tf.int32, shape=[V], name='row_mask')
    prior = tf.constant(0.5, dtype=tf.float32, name='prior')

    # Sufficient statistics are maintained over the complete graph.
    vert_ss = tf.Variable(tf.zeros([V, M], tf.int32), name='vert_ss')
    edge_ss = tf.Variable(tf.zeros([K, M, M], tf.int32), name='edge_ss')
    feat_ss = tf.Variable(tf.zeros([V, C, M], tf.int32), name='feat_ss')

    # Non-normalized probabilities are maintained over the tree.
    vert_probs = tf.Variable(
        tf.cast(vert_ss.initial_value, tf.float32) + prior)
    edge_probs = tf.Variable(
        tf.cast(
            tf.gather(tf.constant(active_edges), edge_ss.initial_value),
            tf.float32) + prior)

    with tf.name_scope('learn'):
        one = tf.constant(1.0, dtype=tf.float32, name='one')
        weights = tf.cast(tf.float32, edge_ss)
        logits = tf.lgamma(weights + prior) - tf.lgamma(weights + one)
        tf.reduce_sum(logits, [1, 2], name='edge_likelihood')

    with tf.name_scope('propagate'):
        schedule = make_propagation_schedule(tree.grid)
        messages = [None] * V
        samples = [None] * V
        with tf.name_scope('feature'):
            counts = tf.gather_nd(feat_ss, tf.stack([tf.range(V), row_data]))
            likelihood = tf.cast(tf.float32, counts) + prior
        with tf.name_scope('inbound'):
            for v, parent, children in reversed(schedule):
                prior_v = vert_probs[v, :]
                message = tf.cond(row_mask[v], likelihood[v] * prior_v,
                                  prior_v)
                for child in children:
                    e = TODO()
                    mat = edge_probs[e, :, :]
                    vec = messages[child, :, tf.newaxis]
                    message *= tf.reduce_sum(mat * vec) / prior_v
                messages[v] = message / tf.reduce_max(message)
        with tf.name_scope('outbound'):
            for v, parent, children in schedule:
                e = TODO()
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
        block = tf.cast(
            tf.gather_nd(vert_probs, vert_indices), dtype=tf.float32) + prior
        vert_probs = tf.scatter_update(vert_probs, vert_indices, block, True)
        block = tf.cast(
            tf.gather_nd(edge_probs, edge_indices), dtype=tf.float32) + prior
        edge_probs = tf.scatter_update(edge_probs, edge_indices, block, True)
        tf.group(
            vert_ss, edge_ss, feat_ss, vert_probs, edge_probs, name='add_row')

    return tf.global_variables_initializer()


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
        self._session = tf.Session()
        self._seed = config['seed']

    def update_session(self):
        self._session.close()
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(self._seed)
            self._seed += 1
            self._init = build_graph(self._structure, self._config)
        self._session = tf.Session(graph=graph)

    def add_row(self, row_id):
        assert row_id not in self._assignments
        fetches = self._session.run(
            ['assignments', 'update/add_row'],
            feed_dict={
                'row_data': self.data[row_id],
                'row_mask': self.mask[row_id],
            })
        self._assignments[row_id] = fetches[0]

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
