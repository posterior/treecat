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

    def build_graph(self):
        '''Builds a tf graph for sampling assignments via message passing.

        Tensors:
          row_data: tf.Tensor of categorical values.
          row_mask: tf.Tensor of presence/absence values for data.
          assignments: Latent mixture class assignments.
          add_row: Target op for adding a row of data.
          remove_row: Target op for removing a row of data.
        '''
        V = self._num_vertices
        M = self._config['num_components']
        C = self._config['num_categories']
        vertex_suffstats = tf.Variable(
            tdype=tf.int32, shape=[V, M, C], name='vertex_suffstats')
        edge_suffstats = tf.Variable(
            dtype=tf.int32, shape=[V, V, M, M], name='edge_suffstats')
        row_data = tf.Placeholder(dtype=tf.int32, shape=[V], name='row_data')
        row_mask = tf.Placeholder(dtype=tf.int32, shape=[V], name='row_mask')
        with tf.name_scope('couple'):
            prior_weights = tf.constant(
                0.5, dtype=tf.float32, name='prior_weights')
            couplings = {}
            for v1, v2 in self._edges:
                counts = tf.slice(edge_suffstats, [v1, v2, 0, 0],
                                  [v1, v2, M, M])
                counts = tf.reshape(counts, [M, M])
                weights = tf.cast(tf.float32, counts) + prior_weights
                couplings[v1, v2] = normalize_rows(weights)
                couplings[v2, v1] = normalize_rows(
                    tf.transpose(weights), [1, 0])
        messages = {}
        samples = {}
        with tf.name_scope('propagate_inbound'):
            for v_in, inbound, _ in reversed(self._schedule):
                counts = tf.slice(vertex_suffstats, [v_in, 0, 0], [v_in, M, C])
                counts = tf.reshape(counts, [M, C])
                weights = tf.cast(tf.float32, counts) + prior_weights
                assert weights and row_data and row_mask  # Pacify flake8.
                probs = TODO('Dirichlet-multinomial')
                for v_out in inbound:
                    probs *= tf.matmul(couplings[v_in, v_out], messages[v_out])
                probs *= tf.reciprocal(tf.reduce_max(probs))
                messages[v_in] = probs
        with tf.name_scope('propagate_outbound'):
            for v_out, _, outbound in self._schedule:
                probs = messages[v_out]
                for v_out in inbound:
                    probs *= tf.matmul(couplings[v_out, v_in], messages[v_in])
                sample = tf.squeeze(tf.multinomial(tf.log(probs), 1), 1)
                probs = tf.one_hot(sample, [M], 1.0, 0.0, dtype=tf.float32)
                messages[v_out] = probs
                samples[v_out] = sample
        samples = [samples[v] for v in range(V)]
        assignments = tf.stack(samples, name='assignments')
        with tf.name_scope('update'):
            vertex_indices = tf.stack([assignments, row_data])
            edge_indices = TODO('stack tuples of the form [v1, v2, m1, m2]')
            add_row_v = tf.scatter_add(vertex_suffstats, edge_indices,
                                       row_mask, True)
            add_row_e = tf.scatter_add(edge_suffstats, vertex_indices,
                                       row_mask, True)
            remove_row_v = tf.scatter_sub(vertex_suffstats, edge_indices,
                                          row_mask, True)
            remove_row_e = tf.scatter_sub(edge_suffstats, vertex_indices,
                                          row_mask, True)
        tf.group(add_row_v, add_row_e, name='add_row')
        tf.group(remove_row_v, remove_row_e, name='remove_row')


def normalize_rows(weights):
    assert len(weights.shape) == 2
    assert weights.shape[0] == weights.shape[1]
    with tf.name_scope('normalize_rows'):
        norms = tf.reshape(tf.reduce_sum(weights, 1), [weights.shape[0], 1])
        return weights * tf.reciprocal(norms)


class Model(object):
    def __init__(self, dataframe, config={}):
        self._config = config
        self._dataframe = dataframe
        self._structure = FeatureTree(config)
        self._assignments = {}  # This maps id -> numpy array.
        self._adding_session = tf.Session()
        self._seed = config['seed']

    def _update_adding_session(self):
        self._adding_session.close()
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(self._seed)
            self._seed += 1
            self._structure().build_graph()
        self._session = tf.Session(graph=graph)

    def add_row(self, row_id):
        assert row_id not in self._assignments
        fetches = self._session.run(
            ['assignments', 'add_row'],
            feed_dict={
                'row_data': self.data[row_id],
                'row_mask': self.mask[row_id]
            })
        self._assignments[row_id] = fetches[0]

    def remove_row(self, row_id):
        assert row_id in self._assignments
        self._session.run(
            'add_row',
            feed_dict={
                'row_data': self._data[row_id],
                'row_mask': self._data[row_id],
                'assignments': self._assignments[row_id]
            })

    def sample_structure(self):
        TODO()
        self._update_adding_session()

    def sample(self):
        '''Sample the entire model using subsample annealed Gibbs sampling.'''
        self._assignments = {}  # Reset assignments.
        TODO('reset sufficient statistics')
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
