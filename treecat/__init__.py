from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import intern

assert np and pd and tf  # Pacify flake8.

DEFAULT_CONFIG = {
    'num_components': 32,
    'num_categories': 3,  # E.g. CSE-IT data.
    'seed': 0,
}


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))


_ACTION_ADD = intern('ACTION_ADD')
_ACTION_REMOVE = intern('ACTION_REMOVE')
_ACTION_STRUCTURE = intern('ACTION_STRUCTURE')


class Variable(object):
    pass


# Component distributions are Dirichlet-categorical.


def suffstats_init(num_vertices, config):
    return np.zeros((num_vertices, config['num_components'],
                     config['num_categories']))


def suffstats_update(suffstats, assignments, row_data, weight=1):
    for v, value in enumerate(row_data):
        if value is not None:
            suffstats[v, assignments[v], value] += weight


class FeatureTree(object):
    def __init__(self, num_vertices, config):
        self._num_vertices = num_vertices
        self._num_components = config['num_components']

        self._suffstats = suffstats_init(num_vertices, config)

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

    Names:
      row_data: Tensor of categorical values.
      row_mask: Tensor of presence/absence values for data.
      assignments: Latent mixture class assignments for a single row.
      update/add_row: Target op for adding a row of data.
      update/remove_row: Target op for removing a row of data.
      learn/edge_likelihood: Likelihoods of edges, used to learn structure.

    Returns:
      op to init global variables.
    '''
    V = tree.num_vertices
    E = tree.num_edges
    M = config['num_components']
    C = config['num_categories']
    row_data = tf.Placeholder(dtype=tf.int32, shape=[V], name='row_data')
    row_mask = tf.Placeholder(dtype=tf.int32, shape=[V], name='row_mask')
    prior = tf.constant(0.5, dtype=tf.float32, name='prior')

    vertex_suffstats = tf.Variable(
        tf.zeros([V, M], tf.int32), dtype=tf.int32, name='vertex_suffstats')
    edge_suffstats = tf.Variable(
        tf.zeros([E, M, M], tf.int32), dtype=tf.int32, name='edge_suffstats')
    feature_suffstats = tf.Variable(
        tf.zeros([V, C, M], tf.int32),
        tdype=tf.int32,
        name='feature_suffstats')

    # These are not normalized.
    vertex_probs = tf.Variable(
        tf.cast(vertex_suffstats.initial_value(), tf.float32) + prior,
        dtype=tf.float32)
    edge_probs = tf.Variable(
        tf.cast(edge_suffstats.initial_value(), tf.float32) + prior,
        dtype=tf.float32)

    with tf.name_scope('propagate'):
        messages = [None] * V
        samples = [None] * V
        with tf.name_scope('feature'):
            counts = tf.gather_nd(feature_suffstats,
                                  tf.stack([tf.range(V), row_data]))
            likelihood = tf.cast(tf.float32, counts) + prior
        with tf.name_scope('inbound'):
            for v_in, inbound, _ in reversed(tree.schedule):
                prior_v = vertex_probs[v_in, :]
                message = tf.cond(row_mask[v_in], likelihood[v_in] * prior_v,
                                  prior_v)
                # TODO tf.cond based on mask.
                for e, v_out in inbound:
                    message *= (tf.reduce_sum(edge_probs[e, :, :] * messages[
                        v_out, :, tf.newaxis]) / prior_v)
                messages[v_in] = message / tf.reduce_max(message)
        with tf.name_scope('outbound'):
            for v_out, _, outbound in tree.schedule:
                message = messages[v_out]
                for e, v_in in outbound:
                    message *= (tf.transpose(
                        tf.reduce_sum(edge_probs[e, :, :], [1, 0]) * messages[
                            v_out, :, tf.newaxis]) / prior_v)
                sample = tf.squeeze(tf.multinomial(tf.log(message), 1), 1)
                message = tf.one_hot(sample, [M], 1.0, 0.0, dtype=tf.float32)
                messages[v_out] = message
                samples[v_out] = sample
    assignments = tf.parallel_stack(samples, name='assignments')

    with tf.name_scope('update'):
        vertex_indices = tf.stack([assignments, row_data])
        edge_indices = TODO('stack tuples of the form [v1, v2, m1, m2]')
        feature_indices = tf.stack([tf.range(V), row_data, assignments])
        row_mask_float = tf.cast(row_mask, tf.float32)
        tf.group(
            tf.scatter_add(vertex_suffstats, vertex_indices, row_mask, True),
            tf.scatter_add(edge_suffstats, edge_indices, row_mask, True),
            tf.scatter_add(feature_suffstats, feature_indices, row_mask, True),
            tf.scatter_add(vertex_probs, vertex_indices, row_mask_float, True),
            tf.scatter_add(edge_probs, edge_indices, row_mask_float, True),
            name='add_row')
        tf.group(
            tf.scatter_sub(vertex_suffstats, vertex_indices, row_mask, True),
            tf.scatter_sub(edge_suffstats, edge_indices, row_mask, True),
            tf.scatter_sub(feature_suffstats, feature_indices, row_mask, True),
            tf.scatter_sub(vertex_probs, vertex_indices, row_mask_float, True),
            tf.scatter_sub(edge_probs, edge_indices, row_mask_float, True),
            name='remove_row')
    with tf.name_scope('learn'):
        one = tf.constant(1.0, dtype=tf.float32, name='one')
        weights = tf.cast(tf.float32, edge_suffstats)
        logits = tf.lgamma(weights + prior) - tf.lgamma(weights + one)
        tf.reduce_sum(logits, [2, 3], name='edge_likelihood')

    return tf.global_variables_initializer()


def matvecmul(numer_mat, denom_vec, arg_vec):
    with tf.name_scope('matvecmul'):
        return tf.reduce_sum(
            tf.multiply(numer_mat, tf.expand_dims(arg_vec, 1))) / denom_vec


class Model(object):
    def __init__(self, dataframe, config={}):
        self._config = config
        self._dataframe = dataframe
        self._structure = FeatureTree(config)
        self._assignments = {}  # This maps id -> numpy array.
        self._session = tf.Session()
        self._seed = config['seed']

    def _update_session(self):
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
            'update/remove_row',
            feed_dict={
                'assignments': self._assignments[row_id],
                'row_data': self._data[row_id],
                'row_mask': self._data[row_id],
            })

    def sample_structure(self):
        TODO()
        self._update_session()

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
