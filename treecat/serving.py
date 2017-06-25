from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from treecat.structure import TreeStructure
from treecat.structure import make_propagation_schedule
from treecat.util import profile

logger = logging.getLogger(__name__)


def make_posterior_factors(grid, suffstats):
    """Computes a complete factorization of the posterior.

    Args:
      suffstats: A dictionary with numpy arrays for sufficient statistics:
        vert_ss, edge_ss, and feat_ss.

    Returns:
      A dictionary with numpy arrays for factors of the posterior:
      observed, observed_latent, latent, latent_latent.
    """
    V, C, M = suffstats['feat_ss'].shape
    E = V - 1
    assert grid.shape == (3, E)
    assert suffstats['vert_ss'].shape == (V, M)
    assert suffstats['edge_ss'].shape == (E, M, M)

    # Hard-code these hyperparameters.
    feat_prior = 0.5  # Jeffreys prior.
    vert_prior = 1.0 / M  # Nonparametric.
    edge_prior = 1.0 / M**2  # Nonparametric.

    # First compute overlapping joint posteriors.
    latent = vert_prior + suffstats['vert_ss'].astype(np.float32)
    latent_latent = edge_prior + suffstats['edge_ss'].astype(np.float32)
    observed_latent = feat_prior + suffstats['feat_ss'].astype(np.float32)
    latent /= latent.sum(1, keepdims=True)
    latent_latent /= latent_latent.sum((1, 2), keepdims=True)
    observed_latent /= observed_latent.sum((1, 2), keepdims=True)

    # Correct observed_latent for partially observed data.
    partial_latent = observed_latent.sum(1)
    observed_latent *= (latent / partial_latent)[:, tf.newaxis, :]

    # Finally compute a non-overlapping factorization.
    observed = observed_latent.sum(2)
    observed_latent /= observed[:, :, tf.newaxis]
    observed_latent /= latent[:, tf.newaxis, :]
    latent_latent /= latent[grid[1, :], :, np.newaxis]
    latent_latent /= latent[grid[2, :], np.newaxis, :]

    return {
        'observed': observed,
        'observed_latent': observed_latent,
        'latent': latent,
        'latent_latent': latent_latent,
    }


@profile
def build_serving_graph(tree, suffstats, config, num_rows):
    """Builds a tensorflow graph for using a trained model.

    This implements two message passing algorithms along the latent tree:
    1. an algorithm to draw conditional samples and
    2. an algorithm to compute conditional log probability.
    The sample algorithm follows a two part inward-outward schedule (that
    generalizes the standard HMM forward-backward algorithms from chains to
    trees). The inward pass is also used by the logprob algorithm.

    Args:
      tree: A TreeStructure object.
      suffstats: A dictionary with numpy arrays for sufficient statistics:
        vert_ss, edge_ss, and feat_ss.
      config: A global config dict.
      num_rows: A fixed number of rows for this graph.

    Names:
      Let N be a dynamic number of rows and V be the number of vertices.
      data: An [N, V] placeholder for input data.
      mask: An [V] placeholder for input presence/absence, same for all rows.
      sample: An [N, V] tensor for imputed output data.
      logprob: An [N] tensor of logprob values of each data row.
    """
    logger.debug('build_serving_graph of tree with %d vertices',
                 tree.num_vertices)
    assert isinstance(tree, TreeStructure)
    assert num_rows > 0
    V = tree.num_vertices
    M = config['num_clusters']  # Clusters in each mixture model.
    C = config['num_categories']  # Categories possible for each feature.
    N = num_rows  # Number of rows in data.
    schedule = make_propagation_schedule(tree.tree_grid)
    factors = make_posterior_factors(tree.tree_grid, suffstats)
    data = tf.placeholder(tf.int32, [N, V], name='data')
    mask = tf.placeholder(tf.bool, [V], name='mask')

    # These posterior factors are constant across calls.
    factor_observed_latent = tf.Variable(factors['observed_latent'])
    factor_latent = tf.Variable(factors['latent'])
    factor_latent_latent = tf.Variable(factors['latent_latent'])

    # Propagate upward from observed to latent.
    # This is run during both sample() and logprob().
    messages = [None] * V
    with tf.name_scope('upward'):
        for v, parent, children in reversed(schedule):

            def present(v=v):
                return tf.multiply(factor_latent[v, tf.newaxis, :],
                                   tf.gather(factor_observed_latent[v, :],
                                             data[:, v]))

            def absent(v=v):
                return tf.tile(factor_latent[v, tf.newaxis, :], [N, 1])

            messages[v] = tf.cond(mask[v], present, absent)
            assert messages[v].shape == [N, M]

    # Propagate latent state inward from children to v.
    # This is run during both sample() and logprob().
    messages_scale = [None] * V
    with tf.name_scope('inward'):
        for v, parent, children in reversed(schedule):
            message = messages[v]
            for child in children:
                e = tree.find_edge(v, child)
                if v < child:
                    trans = factor_latent_latent[e, :, :]
                else:
                    trans = tf.transpose(factor_latent_latent[e, :, :], [1, 0])
                message *= tf.matmul(messages[child], trans)
                assert message.shape == [N, M]
            messages_scale[v] = tf.reduce_max(message)
            messages[v] = message / messages_scale[v]

    # Aggregate the total logprob.
    # This is run only during logprob().
    root, parent, children = schedule[0]
    assert parent is None
    logprob = tf.add(
        tf.log(tf.reduce_sum(messages[root], axis=1)),
        tf.reduce_sum(tf.log(tf.parallel_stack(messages_scale))),
        name='logprob')
    assert logprob.shape == [N]

    # Propagate latent state outward from parent to v.
    # This is run only during the sample() function.
    with tf.name_scope('outward'):
        latent_samples = [None] * V
        for v, parent, children in schedule:
            message = messages[v]
            if parent is not None:
                e = tree.find_edge(v, parent)
                if parent < v:
                    trans = factor_latent_latent[e, :, :]
                else:
                    trans = tf.transpose(factor_latent_latent[e, :, :], [1, 0])
                message *= tf.gather(trans, latent_samples[parent])[0, :]
            assert message.shape == [N, M]
            latent_samples[v] = tf.cast(
                tf.multinomial(tf.log(message), 1)[:, 0], tf.int32)
            assert latent_samples[v].shape == [N]

    # Propagate downward from latent to observed.
    # This is run only during sample().
    with tf.name_scope('downward'):
        observed_samples = [None] * V
        for v, parent, children in schedule:

            def copied(v=v):
                return data[:, v]

            def sampled(v=v):
                logits = tf.log(
                    tf.gather(
                        tf.transpose(factor_observed_latent[v, :, :], [1, 0]),
                        latent_samples[v]))
                assert logits.shape == [N, C]
                return tf.cast(tf.multinomial(logits, 1)[:, 0], tf.int32)

            observed_samples[v] = tf.cond(mask[v], copied, sampled)
            assert observed_samples[v].shape == [N]
    sample = tf.transpose(
        tf.parallel_stack(observed_samples), [1, 0], name='sample')
    assert sample.shape == [N, V]


class ServerBase(object):
    """Base class for serving queries against a trained TreeCat model."""
    pass


class TreeCatServer(ServerBase):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, tree, suffstats, config):
        logger.info('TreeCatServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        self._tree = tree
        self._suffstats = suffstats
        self._config = config
        self._seed = config['seed']
        self._session = None
        self._num_rows = None

    def __del__(self):
        if self._session is not None:
            self._session.close()

    def gc(self):
        """Garbage collect temporary resources."""
        if self._session is not None:
            self._session.close()
            self._session = None
            self._num_rows = None

    # This works around an issue with tf.multinomial and dynamic sizing.
    # Each time a different size is requested, a new graph is built.
    def _get_session(self, num_rows):
        assert num_rows is not None
        if self._num_rows != num_rows:
            self.gc()
            with tf.Graph().as_default():
                tf.set_random_seed(self._seed)
                build_serving_graph(self._tree, self._suffstats, self._config,
                                    num_rows)
                init = tf.global_variables_initializer()
                self._session = tf.Session()
            self._session.run(init)
            self._num_rows = num_rows
        return self._session

    @profile
    def sample(self, data, mask):
        """Sample from the posterior conditional distribution.

        Let V be the number of features and N be the number of rows in input
        data. This function draws in parallel N samples, each sample
        conditioned on one of the input data rows.

        Args:
          data: An [N, V] numpy array of data on which to condition.
            Masked data values should be set to 0.
          mask: An [V] numpy array of presence/absence of conditioning data.
            The mask is constant across rows.
            To sample from the unconditional posterior, set mask to all False.

        Returns:
          An [N, V] numpy array of sampled data, where the cells that where
          conditioned on should match the value of input data, and the cells
          that were not present should be randomly sampled from the conditional
          posterior.
        """
        logger.info('sampling %d rows', data.shape[0])
        num_rows = data.shape[0]
        assert data.shape[1] == self._tree.num_vertices
        assert mask.shape[0] == self._tree.num_vertices
        session = self._get_session(num_rows)
        result = session.run('sample:0', {'data:0': data, 'mask:0': mask})
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
        num_rows = data.shape[0]
        assert data.shape[1] == self._tree.num_vertices
        assert mask.shape[0] == self._tree.num_vertices
        session = self._get_session(num_rows)
        result = session.run('logprob:0', {'data:0': data, 'mask:0': mask})
        assert result.shape == (num_rows, )
        return result
