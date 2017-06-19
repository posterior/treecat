from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from treecat.structure import TreeStructure
from treecat.structure import make_propagation_schedule
from treecat.util import COUNTERS
from treecat.util import profile
from treecat.util import sizeof

logger = logging.getLogger(__name__)


def build_graph(tree, suffstats, config):
    """Builds a tensorflow graph for using a trained model.

    Args:
      tree: A TreeStructure object.
      suffstats: A dictionary with numpy arrays for sufficient statistics:
        vert_ss, edge_ss, and feat_ss.
      config: A global config dict.

    Names:
      Let N be a dynamic number of rows and V be the number of vertices.
      data: An [N, V] placeholder for input data.
      mask: An [N, V] placeholder for input presence/absence.
      sample: An [N, V] tensor for imputed output data.
      logprob: An [N] tensor of logprob values of each data row.
    """
    assert isinstance(tree, TreeStructure)
    logger.debug('build_graph of tree with %d vertices' % tree.num_vertices)
    V = tree.num_vertices
    E = V - 1  # Number of edges in the tree
    M = config['num_clusters']  # Clusters in each mixture model.
    C = config['num_categories']  # Categories possible for each feature..
    N = None  # Number of rows in data.
    assert suffstats['vert_ss'].shape == [V, M]
    assert suffstats['edge_ss'].shape == [E, M, M]
    assert suffstats['feat_ss'].shape == [V, C, M]
    schedule = make_propagation_schedule(tree.tree_grid)
    vertices = tf.range(V, dtype=np.int32)

    # Hard-code these hyperparameters.
    feat_prior = 0.5  # Jeffreys prior.
    vert_prior = 1.0 / M  # Nonparametric.
    edge_prior = 1.0 / M**2  # Nonparametric.

    # Non-normalized probabilities are constant but shared across queries.
    feat_probs = tf.Variable(
        feat_prior + suffstats['feat_ss'].astype(np.float32),
        name='feat_probs')
    vert_probs = tf.Variable(
        vert_prior + suffstats['vert_ss'].astype(np.float32),
        name='vert_probs')
    edge_probs = tf.Variable(
        edge_prior + suffstats['edge_ss'].astype(np.float32),
        name='edge_probs')

    COUNTERS.footprint_serving_feat_probs = sizeof(feat_probs)
    COUNTERS.footprint_serving_vert_probs = sizeof(vert_probs)
    COUNTERS.footprint_serving_edge_probs = sizeof(edge_probs)

    # These are the inputs to both sample() and logprob().
    data = tf.placeholder(tf.int32, [N, V], name='data')
    mask = tf.placeholder(tf.bool, [N, V], name='mask')

    with tf.name_scope('inbound'):
        messages = [None] * V
        with tf.name_scope('observed'):
            indices = tf.stack([vertices, data], 1)
            likelihood = tf.gather_nd(feat_probs, indices)
        with tf.name_scope('latent'):
            for v, parent, children in reversed(schedule):
                prior_v = vert_probs[v, :]
                message = tf.cond(mask[v], lambda: prior_v,
                                  lambda: likelihood[v] * prior_v)
                for child in children:
                    e = tree.find_edge(v, child)
                    mat = edge_probs[e, :, :]
                    vec = messages[child][:, tf.newaxis]
                    message *= tf.reduce_sum(mat * vec) / prior_v
                messages[v] = message / tf.reduce_max(message)
    with tf.name_scope('sample'):
        samples = [None] * V
        with tf.name_scope('latent'):
            for v, parent, children in schedule:
                message = messages[v]
                if parent is not None:
                    e = tree.find_edge(v, parent)
                    prior_v = vert_probs[v, :]
                    mat = tf.transpose(edge_probs[e, :, :], [1, 0])
                    message *= tf.gather(mat, samples[parent])[0, :] / prior_v
                    assert len(message.shape) == 1
                sample = tf.cast(
                    tf.multinomial(tf.log(message)[tf.newaxis, :], 1)[0],
                    tf.int32)
                samples[v] = sample
    # TODO Add graph for s


class TreeCatServer(object):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, tree, suffstats, config):
        logger.info('TreeCatServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        self._tree = tree
        self._config = config
        self._seed = config
        with tf.Graph().as_default():
            tf.set_random_seed(self._seed)
            init = tf.global_variables_initializer()
            build_graph(tree, suffstats, self._config)
            self._session = tf.Session()
        self._session.run(init)

    @profile
    def sample(self, data, mask):
        """Sample from the posterior conditional distribution.

        Let V be the number of features and N be the number of rows in input
        data and mask. This function draws in parallel N samples, each sample
        conditioned on one of the input data rows.

        Args:
          data: An [N, V] numpy array of data on which to condition.
            Masked data values should be set to 0.
          mask: An [N, V] numpy array of presence/absence of conditioning data.
            To sample from the unconditional posterior, set mask to all False.

        Returns:
          An [N, V] numpy array of sampled data, where the cells that where
          conditioned on should match the value of input data, and the cells
          that were not present should be randomly sampled from the conditional
          posterior.
        """
        logger.info('sampling %d rows', data.shape[0])
        assert data.shape == mask.shape
        assert data.shape[1] == self._tree.num_vertices
        result = self._sess.run('sample', {'data': data, 'mask': mask})
        assert result.shape == data.shape
        assert result.dtype == data.dtype
        return result

    @profile
    def logprob(self, data, mask):
        """Compute log probabilities of each row of data.

        Let V be the number of features and N be the number of rows in input
        data and mask. This function computes in parallel the logprob of each
        of the N input data rows.

        Args:
          data: An [N, V] numpy array of data on which to condition.
            Masked data values should be set to 0.
          mask: An [N, V] numpy array of presence/absence of conditioning data.

        Returns:
          An [N] numpy array of log probabilities.
        """
        logger.info('computing logprob of %d rows', data.shape[0])
        assert data.shape == mask.shape
        assert data.shape[1] == self._tree.num_vertices
        result = self._sess.run('logprob', {'data': data, 'mask': mask})
        assert result.shape == [data.shape[0]]
        return result
