from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from treecat.serving import ServerBase
from treecat.serving import make_posterior_factors
from treecat.structure import TreeStructure
from treecat.structure import find_complete_edge
from treecat.structure import make_propagation_schedule
from treecat.structure import sample_tree
from treecat.training import TrainerBase
from treecat.util import COUNTERS
from treecat.util import profile
from treecat.util import sizeof

logger = logging.getLogger(__name__)


def tf_matvecmul(matrix, vector):
    assert len(matrix.shape) == 2
    assert len(vector.shape) == 1
    assert matrix.shape[1] == vector.shape[0]
    return tf.reduce_sum(matrix * vector[:, tf.newaxis], axis=1)


@profile
def build_training_graph(tree, inits, config):
    """Builds a tensorflow graph for sampling assignments via message passing.

    Feature distributions are Dirichlet-categorical.
    TODO Switch to Beta-binomial or Beta-(Poisson Binomial).

    Args:
      tree: A TreeStructure object.
      inits: An dict of optional saved tensor values.
      config: A global config dict.

    Names:
      row_data: Tensor of categorical values.
      row_mask: Tensor of presence/absence values for data.
        Should be positive to add a row and negative to remove a row.
      assignments: Latent mixture class assignments for a single row.
      assign/add_row: Target op for adding a row of data.
      assign/remove_row: Target op for removing a row of data.
      structure/edge_logits: Non-normalized log probabilities of edges.

    Returns:
      A dictionary of actions whose values can be input to Session.run():
        load: Initialize global variables.
        save: Eval global variables.
    """
    logger.debug('build_training_graph of tree with %d vertices',
                 tree.num_vertices)
    assert isinstance(tree, TreeStructure)
    assert isinstance(inits, dict)
    assert isinstance(config, dict)
    V = tree.num_vertices
    E = V - 1  # Number of edges in the tree.
    K = V * (V - 1) // 2  # Number of edges in the complete graph.
    M = config['num_clusters']  # Clusters in each mixture model.
    C = config['num_categories']  # Categories possible for each feature.
    vertices = tf.range(V, dtype=tf.int32)
    tree_grid = tf.constant(tree.tree_grid)
    complete_grid = tf.constant(tree.complete_grid)
    schedule = make_propagation_schedule(tree.tree_grid)
    row_data = tf.placeholder(dtype=tf.int32, shape=[V], name='row_data')
    row_mask = tf.placeholder(dtype=tf.bool, shape=[V], name='row_mask')

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

    COUNTERS.footprint_training_vert_ss = sizeof(vert_ss)
    COUNTERS.footprint_training_edge_ss = sizeof(edge_ss)
    COUNTERS.footprint_training_feat_ss = sizeof(feat_ss)
    COUNTERS.footprint_training_tree_ss = sizeof(tree_ss)
    COUNTERS.footprint_training_vert_probs = sizeof(vert_probs)
    COUNTERS.footprint_training_edge_probs = sizeof(edge_probs)

    # This copies tree_ss to edge_ss after updating the tree structure.
    edges = tf.placeholder(dtype=tf.int32, shape=[E], name='edges')
    tf.assign(edge_ss, tf.gather(tree_ss, edges), name='update_tree')

    with tf.name_scope('structure'):
        # This is run to compute edge logits for learning the tree structure.
        one = tf.constant(1.0, dtype=tf.float32, name='one')
        weights = tf.cast(tree_ss, tf.float32)
        logits = tf.lgamma(weights + edge_prior) - tf.lgamma(weights + one)
        tf.reduce_sum(logits, [1, 2], name='edge_logits')

    # Propagate upward from observed to latent.
    # This is run only during add_row().
    with tf.name_scope('upward'):
        indices = tf.stack([vertices, row_data], 1)
        counts = tf.gather_nd(feat_ss, indices)
        data_probs = feat_prior + tf.cast(counts, tf.float32)
        assert data_probs.shape == [V, M]

    # Propagate latent state inward from children to v.
    # This is run only during add_row().
    with tf.name_scope('inward'):
        messages = [None] * V
        for v, parent, children in reversed(schedule):
            prior_v = vert_probs[v, :]
            assert prior_v.shape == [M]

            def present(prior_v=prior_v, v=v):
                return prior_v * data_probs[v, :]

            def absent(prior_v=prior_v):
                return prior_v

            message = tf.cond(row_mask[v], present, absent)
            assert message.shape == [M]
            for child in children:
                e = tree.find_tree_edge(v, child)
                if v < child:
                    trans = edge_probs[e, :, :]
                else:
                    trans = tf.transpose(edge_probs[e, :, :], [1, 0])
                message *= tf_matvecmul(trans, messages[child]) / prior_v
            messages[v] = message / tf.reduce_max(message)

    # Propagate latent state outward from parent to v.
    # This is run only during add_row().
    with tf.name_scope('outward'):
        latent_samples = [None] * V
        for v, parent, children in schedule:
            message = messages[v]
            if parent is not None:
                e = tree.find_tree_edge(v, parent)
                prior_v = vert_probs[v, :]
                if parent < v:
                    trans = edge_probs[e, :, :]
                else:
                    trans = tf.transpose(edge_probs[e, :, :], [1, 0])
                message *= tf.gather(trans,
                                     latent_samples[parent])[0, :] / prior_v
                assert message.shape == [M]
            latent_samples[v] = tf.cast(
                tf.multinomial(tf.log(message)[tf.newaxis, :], 1)[0], tf.int32)

    assignments = tf.squeeze(
        tf.parallel_stack(latent_samples), name='assignments')
    with tf.control_dependencies([tf.assert_less(assignments, M)]):
        assignments = tf.identity(assignments)

    # This suffstats update is run during add_row() and remove_row().
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
        # Updates are adapted to shape and dtype.
        d_feat_ss = tf.cast(row_mask, tf.int32)
        d_vert_ss = tf.ones([V], tf.int32)
        d_edge_ss = tf.ones([E], tf.int32)
        d_tree_ss = tf.ones([K], tf.int32)
        d_vert_probs = tf.ones([V], tf.float32)
        d_edge_probs = tf.ones([E], tf.float32)
        tf.group(
            tf.scatter_nd_add(feat_ss, feat_indices, d_feat_ss, True),
            tf.scatter_nd_add(vert_ss, vert_indices, d_vert_ss, True),
            tf.scatter_nd_add(edge_ss, edge_indices, d_edge_ss, True),
            tf.scatter_nd_add(tree_ss, tree_indices, d_tree_ss, True),
            tf.scatter_nd_add(vert_probs, vert_indices, d_vert_probs, True),
            tf.scatter_nd_add(edge_probs, edge_indices, d_edge_probs, True),
            name='add_row')
        tf.group(
            tf.scatter_nd_sub(feat_ss, feat_indices, d_feat_ss, True),
            tf.scatter_nd_sub(vert_ss, vert_indices, d_vert_ss, True),
            tf.scatter_nd_sub(edge_ss, edge_indices, d_edge_ss, True),
            # Note that tree_ss is not updated during remove_row.
            tf.scatter_nd_sub(vert_probs, vert_indices, d_vert_probs, True),
            tf.scatter_nd_sub(edge_probs, edge_indices, d_edge_probs, True),
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


class TensorflowTrainer(TrainerBase):
    """Class for training a TreeCat model using Tensorflow."""

    def __init__(self, data, mask, config):
        """Initialize a model in an unassigned state.

        Args:
            data: A 2D array of categorical data.
            mask: A 2D array of presence/absence, where present = True.
            config: A global config dict.
        """
        logger.info('TensorflowTrainer of %d x %d data', data.shape[0],
                    data.shape[1])
        super(TensorflowTrainer, self).__init__(data, mask, config)
        self._session = None
        self._update_tree()

    def __del__(self):
        if self._session is not None:
            self._session.close()

    @profile
    def _update_tree(self):
        if self._session is not None:
            edges = [
                find_complete_edge(v1, v2)
                for e, v1, v2 in self.tree.tree_grid.T
            ]
            self._session.run('update_tree', {'edges:0': edges})
            self.suffstats = self._session.run(self._actions['save'])
            self._session.close()
        with tf.Graph().as_default():
            tf.set_random_seed(self._seed)
            self._actions = build_training_graph(self.tree, self.suffstats,
                                                 self._config)
            self._session = tf.Session()
        self._session.run(self._actions['load'])

    @profile
    def add_row(self, row_id):
        logger.debug('TensorflowTrainer.add_row %d', row_id)
        assert row_id not in self._assigned_rows, row_id
        self._assigned_rows.add(row_id)
        assignments, _ = self._session.run(
            ['assignments:0', 'update/add_row'],
            feed_dict={
                'row_data:0': self._data[row_id],
                'row_mask:0': self._mask[row_id],
            })
        assert assignments.shape == (self._data.shape[1], )
        self.assignments[row_id, :] = assignments

    @profile
    def remove_row(self, row_id):
        logger.debug('TensorflowTrainer.remove_row %d', row_id)
        assert row_id in self._assigned_rows, row_id
        self._assigned_rows.remove(row_id)
        self._session.run(
            'update/add_row',
            feed_dict={
                'assignments:0': self.assignments[row_id],
                'row_data:0': self._data[row_id],
                'row_mask:0': self._mask[row_id],
            })

    @profile
    def sample_tree(self):
        logger.info('TensorflowTrainer.sample_tree given %d rows',
                    len(self._assigned_rows))
        edge_logits = self._session.run('structure/edge_logits:0')
        complete_grid = self.tree.complete_grid
        assert edge_logits.shape[0] == complete_grid.shape[1]
        edges = self.tree.tree_grid[1:3, :].T
        edges = sample_tree(
            complete_grid,
            edge_logits,
            edges,
            seed=self._seed,
            steps=self._config['sample_tree_steps'])
        self._seed += 1
        self.tree.set_edges(edges)
        self._update_tree()

    def finish(self):
        logger.info('TensorflowTrainer.finish with %d rows',
                    len(self._assigned_rows))
        self.suffstats = self._session.run(self._actions['save'])
        self.tree.gc()
        self._session.close()
        self._session = None


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
                e = tree.find_tree_edge(child, v)
                if child < v:
                    trans = factor_latent_latent[e, :, :]
                else:
                    trans = tf.transpose(factor_latent_latent[e, :, :], [1, 0])
                message *= tf.matmul(messages[child], trans)
                assert message.shape == [N, M]
            messages_scale[v] = tf.reduce_max(message)
            messages[v] = message / messages_scale[v]

    # Aggregate the total logprob.
    # This is run only during logprob().
    # FIXME This does not account for feature_observed.
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
                e = tree.find_tree_edge(parent, v)
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


class TensorflowServer(ServerBase):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, tree, suffstats, config):
        logger.info('TensorflowServer with %d features', tree.num_vertices)
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
        logger.info('computing logprob of %d rows', data.shape[0])
        num_rows = data.shape[0]
        assert data.shape[1] == self._tree.num_vertices
        assert mask.shape[0] == self._tree.num_vertices
        session = self._get_session(num_rows)
        result = session.run('logprob:0', {'data:0': data, 'mask:0': mask})
        assert result.shape == (num_rows, )
        return result
