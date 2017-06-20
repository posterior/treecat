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


def make_logits(prior, suffstats):
    '''Convert a conjugate prior + suffstats to normalized logits.

    Args:
      prior: A prior with shape broadcastable to suffstats.
      suffstats: A numpy array of sufficient statistics (counts).

    Returns:
      A numpy array of logits. The exp() of will be L1 normalized for each
      leading index, i.e. np.allclose(1, np.exp(result).sum(axis=0)).
    '''
    probs = prior + suffstats.astype(np.float32)
    totals = probs.sum(axis=0)
    assert (totals > 0).all()
    probs /= totals.reshape((1, ) + probs.shape[1:])
    return np.log(probs)


@profile
def build_graph(tree, suffstats, config, num_rows):
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
    logger.debug('build_graph of tree with %d vertices' % tree.num_vertices)
    assert isinstance(tree, TreeStructure)
    assert num_rows > 0
    V = tree.num_vertices
    E = V - 1  # Number of edges in the tree
    M = config['num_clusters']  # Clusters in each mixture model.
    C = config['num_categories']  # Categories possible for each feature..
    N = num_rows  # Number of rows in data.
    assert suffstats['vert_ss'].shape == (V, M)
    assert suffstats['edge_ss'].shape == (E, M, M)
    assert suffstats['feat_ss'].shape == (V, C, M)
    schedule = make_propagation_schedule(tree.tree_grid)

    # Hard-code these hyperparameters.
    feat_prior = 0.5  # Jeffreys prior.
    vert_prior = 1.0 / M  # Nonparametric.
    edge_prior = 1.0 / M**2  # Nonparametric.

    # These log probabilities are constant but shared across queries.
    feat_logits = tf.Variable(make_logits(feat_prior, suffstats['feat_ss']))
    vert_logits = tf.Variable(make_logits(vert_prior, suffstats['vert_ss']))
    edge_logits = tf.Variable(make_logits(edge_prior, suffstats['edge_ss']))

    COUNTERS.footprint_serving_feat_logits = sizeof(feat_logits)
    COUNTERS.footprint_serving_vert_logits = sizeof(vert_logits)
    COUNTERS.footprint_serving_edge_logits = sizeof(edge_logits)

    # These are the inputs to both sample() and logprob().
    data = tf.placeholder(tf.int32, [N, V], name='data')
    mask = tf.placeholder(tf.bool, [V], name='mask')

    # This inward pass is run during both sample() and logprob() functions.
    with tf.name_scope('inward'):
        messages = [None] * V  # Properly scaled log probabilities.
        with tf.name_scope('observed'):
            vertices = tf.zeros_like(data)
            vertices += tf.range(V, dtype=np.int32)[tf.newaxis, :]
            indices = tf.stack([vertices, data], 2)
            data_logits = tf.gather_nd(feat_logits, indices)
            assert data_logits.shape == [N, V, M]
        with tf.name_scope('latent'):
            for v, parent, children in reversed(schedule):
                message = tf.cond(
                    mask[v],
                    lambda: tf.zeros([N, M]),
                    lambda v=v: data_logits[:, v, :])
                assert message.shape == [N, M]
                for child in children:
                    e = tree.find_edge(v, child)
                    # This is a matrix-vector multiply in log space.
                    message += tf.reduce_logsumexp(
                        edge_logits[e, :, :] + messages[child][:, tf.newaxis])
                prior_v = vert_logits[tf.newaxis, v, :]
                assert prior_v.shape == [1, M]
                messages[v] = message + (1.0 - len(children)) * prior_v
    root, parent, children = schedule[0]
    assert parent is None
    logprob = tf.reduce_logsumexp(messages[root], axis=1, name='logprob')
    assert logprob.shape == [N]

    # This outward pass is run only during the sample() function.
    with tf.name_scope('outward'):
        latent_samples = [None] * V
        observed_samples = [None] * V
        with tf.name_scope('latent'):
            for v, parent, children in schedule:
                message = messages[v]
                if parent is not None:
                    e = tree.find_edge(v, parent)
                    # This is a matrix multiply with a one-hot vector.
                    message += tf.gather(
                        tf.transpose(edge_logits[e, :, :], [1, 0]),
                        latent_samples[parent])[0, :]
                    # FIXME Subtract off previous inward message.
                assert message.shape == [N, M]
                latent_samples[v] = tf.cast(
                    tf.multinomial(message, 1)[:, 0], tf.int32)
                assert latent_samples[v].shape == [N]
        with tf.name_scope('observed'):
            for v, parent, children in schedule:
                assert feat_logits.shape == [V, C, M]

                def copied(v=v):
                    return data[:, v]

                def sampled(v=v):
                    logits = tf.gather(
                        tf.transpose(feat_logits[v, :, :], [1, 0]),
                        latent_samples[v])
                    assert logits.shape == [N, C]
                    return tf.cast(tf.multinomial(logits, 1)[:, 0], tf.int32)

                observed_samples[v] = tf.cond(mask[v], copied, sampled)
                assert observed_samples[v].shape == [N]
    sample = tf.transpose(
        tf.parallel_stack(observed_samples), [1, 0], name='sample')
    assert sample.shape == [N, V]


class TreeCatServer(object):
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
                build_graph(self._tree, self._suffstats, self._config,
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
        assert result.shape == [num_rows]
        return result
