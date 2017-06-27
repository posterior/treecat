from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from treecat.serving import ServerBase
from treecat.serving import make_posterior_factors
from treecat.structure import TreeStructure
from treecat.structure import make_propagation_schedule
from treecat.util import profile

logger = logging.getLogger(__name__)


def sample_from_probs2(probs, out):
    """Vectorized sampler from categorical distribution."""
    # Adapted from https://stackoverflow.com/questions/40474436
    assert len(probs.shape) == 2
    u = np.random.rand(probs.shape[0], 1)
    cdf = probs.cumsum(axis=1)
    (u < cdf).argmax(axis=1, out=out)


class NumpyServer(ServerBase):
    def __init__(self, tree, suffstats, config):
        logger.info('NumpyServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        self._tree = tree
        self._config = config
        self._seed = config['seed']
        self._factors = make_posterior_factors(tree.tree_grid, suffstats)
        self._schedule = make_propagation_schedule(tree.tree_grid)

        # These are useful dimensions to import into locals().
        V = self._tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        M = self._config[
            'model_num_clusters']  # Clusters in each mixture model.
        C = self._config[
            'model_num_categories']  # Categories for each feature.
        self._VEMC = (V, E, M, C)

    @profile
    def _propagate(self, data, mask, task):
        N = data.shape[0]
        V, E, M, C = self._VEMC
        factor_observed = self._factors['observed']
        factor_observed_latent = self._factors['observed_latent']
        factor_latent = self._factors['latent']
        factor_latent_latent = self._factors['latent_latent']

        messages = np.zeros([V, N, M], dtype=np.float32)
        messages[...] = factor_latent[:, np.newaxis, :]
        logprob = np.zeros([N], dtype=np.float32)
        for v, parent, children in reversed(self._schedule):
            message = messages[v, :, :]
            # Propagate upward from observed to latent.
            if mask[v]:
                message *= factor_observed_latent[v, data[:, v], :]
                logprob += np.log(factor_observed[v, data[:, v]])
            # Propagate latent state inward from children to v.
            for child in children:
                e = self._tree.find_tree_edge(child, v)
                if child < v:
                    trans = factor_latent_latent[e, :, :]
                else:
                    trans = factor_latent_latent[e, :, :].T
                message *= np.dot(messages[child, :, :], trans)  # Expensive.
            message_scale = message.max(axis=1)  # Surprisingly expensive.
            message /= message_scale[:, np.newaxis]
            logprob += np.log(message_scale)

        if task == 'logprob':
            # Aggregate the total logprob.
            root, parent, children = self._schedule[0]
            assert parent is None
            logprob += np.log(messages[root, :, :].sum(axis=1))
            return logprob

        elif task == 'sample':
            latent_samples = np.zeros([V, N], np.int32)
            observed_samples = data.copy()
            for v, parent, children in self._schedule:
                message = messages[v, :, :]
                # Propagate latent state outward from parent to v.
                if parent is not None:
                    e = self._tree.find_tree_edge(parent, v)
                    if parent < v:
                        trans = factor_latent_latent[e, :, :]
                    else:
                        trans = factor_latent_latent[e, :, :].T
                    message *= trans[latent_samples[parent, :], :]
                sample_from_probs2(message, out=latent_samples[v, :])
                # Propagate downward from latent to observed.
                if not mask[v]:
                    probs = factor_observed_latent[v, :, latent_samples[v, :]]
                    assert probs.shape == (N, C)
                    probs /= probs.sum(axis=1, keepdims=True)
                    sample_from_probs2(probs, out=observed_samples[:, v])
            return observed_samples

        raise ValueError('Unknown task: {}'.format(task))

    @profile
    def sample(self, data, mask):
        logger.info('sampling %d rows', data.shape[0])
        N = data.shape[0]
        V, E, M, C = self._VEMC
        assert data.shape == (N, V)
        assert mask.shape == (V, )
        return self._propagate(data, mask, 'sample')

    @profile
    def logprob(self, data, mask):
        logger.debug('computing logprob of %d rows', data.shape[0])
        N = data.shape[0]
        V, E, M, C = self._VEMC
        assert data.shape == (N, V)
        assert mask.shape == (V, )
        return self._propagate(data, mask, 'logprob')
