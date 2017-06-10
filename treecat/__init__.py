from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import intern

assert np and pd and tf  # Pacify pyflakes.

DEFAULT_CONFIG = {
    'num_categories': 16,
}


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))


_ACTION_ADD = intern('ACTION_ADD')
_ACTION_REMOVE = intern('ACTION_REMOVE')
_ACTION_STRUCTURE = intern('ACTION_STRUCTURE')


class Variable(object):
    pass


class FeatureTree(object):
    def __init__(self, num_vertices, config):
        self._num_vertices = num_vertices
        self._num_categories = config['num_categories']

        # Topological structure is initialized arbitrarily.
        # TODO Add confusion matrices to edges.
        self._edges = [(i, i + 1) for i in range(num_vertices - 1)]
        self._neighbors = TODO()

    def build_adding_graph(self):
        '''Builds a tensorflow graph for belief propagation.'''
        graph = tf.Graph()
        with graph.as_default():
            for v in range(self._num_vertices):
                TODO('Add placeholder for observed feature')
            # This assumes edges are topologically sorted with root first.
            for e in reversed(self._edges):
                TODO('Add a tensor for inbound scoring messages')
            for e in self._edges:
                TODO('Add a tensor for outbound sampling messages')
            TODO('Add a single node that aggregates vertex assignments')
            assignments = None
        return graph, assignments


class Model(object):
    def __init__(self, dataframe, config={}):
        self._config = config
        self._dataframe = dataframe
        self._structure = FeatureTree(config)
        self._assignments = {}  # maps id -> numpy array.
        self._adding_session = tf.Session()

    def _update_adding_session(self):
        self._adding_session.close()
        graph, assignments = self._structure().build_adding_graph()
        self._adding_session = tf.Session(graph=graph)
        self._adding_assignments = assignments

    def add_row(self, row_id):
        assert row_id not in self._assignments
        feed_dict = {}  # TODO populate this with observation data.
        self._assignments[row_id] = self._adding_session.run(
            self._adding_assignments, feed_dict)

    def remove_row(self, row_id):
        assert row_id in self._assignments

    def sample_structure(self):
        TODO()
        self._adding_graph = self._structure().build_adding_graph()

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
