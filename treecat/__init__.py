from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import intern

assert np and pd and tf  # Pacify pyflakes.

DEFAULT_CONFIG = {
    'num_components': 32,
    'num_categories': 3,  # E.g. CSE-IT data.
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
        '''Arbitrarily-rooted depth-first topological sort.'''
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
        TODO('create a fast index')

    def init_topology(self):
        # Topological structure is initialized arbitrarily.
        # TODO Add confusion matrices to edges.
        self._edges = [(i, i + 1) for i in range(num_vertices - 1)]
        self._neighbors = TODO()

    def sample_topology(self):
        TODO()

    def build_graph(self):
        '''Builds a tensorflow graph for belief propagation.'''
        graph = tf.Graph()
        with graph.as_default():
            suffstats = tf.Variable(
                tdype=tf.int32,
                shape=(self._num_vertices, self.config['num_components'],
                       self.config['num_categories']))
            row_data = tf.Placeholder(
                dtype=tf.int32, shape=[self._num_vertices], name='row_data')
            # This assumes edges are topologically sorted with root first.
            for e in reversed(self._edges):
                TODO('Add a tensor for inbound scoring messages')
            for e in self._edges:
                TODO('Add a tensor for outbound sampling messages')
            TODO('Add a single node that aggregates vertex assignments')
            assignments = tf.stack([TODO()])
            indices = tf.stack([assignments, row_data])
            ones = tf.ones(suffstats.shape, suffstats.dtype)
            tf.scatter_add(suffstats, indices, ones, True, name='add_row')
            tf.scatter_sub(suffstats, indices, ones, True, name='remove_row')
        return graph


class Model(object):
    def __init__(self, dataframe, config={}):
        self._config = config
        self._dataframe = dataframe
        self._structure = FeatureTree(config)
        self._assignments = {}  # This maps id -> numpy array.
        self._adding_session = tf.Session()

    def _update_adding_session(self):
        self._adding_session.close()
        graph = self._structure().build_graph()
        assignments = graph.get_operation_by_name('assignments')
        self._session = tf.Session(graph=graph)

    def add_row(self, row_id):
        assert row_id not in self._assignments
        feed_dict = {}  # TODO populate this with observation data.
        fetches = self._session.run(
            [
                self._session.graph.get_operation_by_name('assignments'),
                self._session.graph.get_operation_by_name('add_row')
            ],
            feed_dict={'observation': self._data[row_id]})
        self._assignments[row_id] = fetches[0]

    def remove_row(self, row_id):
        assert row_id in self._assignments
        self._session.run(
            self._session.graph.get_operation_by_name('add_row'),
            feed_dict={
                'observation': self._data[row_id],
                'assignments': self._assignments[row_id]
            })

    def sample_structure(self):
        TODO()
        self._adding_graph = self._structure().build_graph()

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
