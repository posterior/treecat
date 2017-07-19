from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy.misc import logsumexp
from scipy.stats import entropy

from six.moves import range
from six.moves import zip
from treecat.format import export_rows
from treecat.format import import_rows
from treecat.format import pickle_load
from treecat.structure import OP_DOWN
from treecat.structure import OP_IN
from treecat.structure import OP_OUT
from treecat.structure import OP_ROOT
from treecat.structure import OP_UP
from treecat.structure import TreeStructure
from treecat.structure import estimate_tree
from treecat.structure import make_propagation_program
from treecat.structure import sample_tree
from treecat.util import SQRT_TINY
from treecat.util import TINY
from treecat.util import guess_counts
from treecat.util import profile
from treecat.util import quantize_from_probs2
from treecat.util import sample_from_probs2

logger = logging.getLogger(__name__)


def correlation(probs):
    """Compute correlation rho(X,Y) = sqrt(1 - exp(-2 I(X;Y))).

    Args:
      probs: An [M, M]-shaped numpy array representing a joint distribution.

    Returns:
      A number in [0,1) representing the information-theoretic correlation.
    """
    assert len(probs.shape) == 2
    assert probs.shape[0] == probs.shape[1]
    mutual_information = (entropy(probs.sum(0)) + entropy(probs.sum(1)) -
                          entropy(probs.flatten()))
    return np.sqrt(1.0 - np.exp(-2.0 * mutual_information))


class ServerBase(object):
    """Base class for TreeCat and Ensemble servers."""

    def __init__(self, ragged_index):
        self._ragged_index = ragged_index
        self._zero_row = np.zeros(self.ragged_size, np.int8)

    @property
    def ragged_index(self):
        return self._ragged_index

    @property
    def ragged_size(self):
        return self._ragged_index[-1]

    def make_zero_row(self):
        """Make an empty data row."""
        return self._zero_row.copy()

    def marginals(self, data):
        raise NotImplementedError

    def median(self, counts, data):
        """Compute L1-loss-minimizing quantized marginals conditioned on data.

        Args:
          counts: A [V]-shaped numpy array of quantization resolutions.
          data: An [N, R]-shaped numpy array of row of conditioning data, as a
            ragged nummpy array of multinomial counts,
            where R = server.ragged_size.

        Returns:
          An array of the same shape as data, but with specified counts.
        """
        logger.debug('computing median')
        R = self.ragged_size
        N = data.shape[0]
        assert data.shape == (N, R)
        assert data.dtype == np.int8

        # Compute marginals (this is the bulk of the work).
        marginals = self.marginals(data)

        # Quantize the marginals.
        V = len(self._ragged_index) - 1
        result = np.zeros_like(data)
        for v in range(V):
            beg, end = self._ragged_index[v:v + 2]
            result[:, beg:end] = quantize_from_probs2(marginals[:, beg:end],
                                                      counts[v])

        return result

    def mode(self, counts, data):
        """Compute a maximum a posteriori data value conditioned on data.

        Args:
          counts: A [V]-shaped numpy array of quantization resolutions.
          data: An [N, R]-shaped numpy array of row of conditioning data, as a
            ragged nummpy array of multinomial counts,
            where R = server.ragged_size.

        Returns:
          An array of the same shape as data, but with specified counts.
        """
        raise NotImplementedError


class TreeCatServer(ServerBase):
    """Class for serving queries against a trained TreeCat model."""

    def __init__(self, model):
        """Create a TreeCat server.

        Args:
          model: A dict with fields:
            tree: A TreeStructure.
            suffstats: A dict of sufficient statistics.
            edge_logits: A K-sized array of nonnormalized edge probabilities.
            config: A global config dict.
        """
        tree = model['tree']
        suffstats = model['suffstats']
        config = model['config']
        logger.info('TreeCatServer with %d features', tree.num_vertices)
        assert isinstance(tree, TreeStructure)
        ragged_index = suffstats['ragged_index']
        ServerBase.__init__(self, ragged_index)
        self._tree = tree
        self._config = config
        self._program = make_propagation_program(tree.tree_grid)

        # These are useful dimensions to import into locals().
        V = self._tree.num_vertices
        E = V - 1  # Number of edges in the tree.
        M = self._config['model_num_clusters']  # Number of latent clusters.
        R = ragged_index[-1]  # Size of ragged data.
        self._VEMR = (V, E, M, R)

        # Use Jeffreys priors.
        vert_prior = 0.5
        edge_prior = 0.5 / M
        feat_prior = 0.5 / M
        meas_prior = feat_prior * np.array(
            [(ragged_index[v + 1] - ragged_index[v]) for v in range(V)],
            dtype=np.float32).reshape((V, 1))

        # These are posterior marginals for vertices and pairs of vertices.
        self._vert_probs = suffstats['vert_ss'].astype(np.float32) + vert_prior
        self._vert_probs /= self._vert_probs.sum(axis=1, keepdims=True)
        self._edge_probs = suffstats['edge_ss'].astype(np.float32) + edge_prior
        self._edge_probs /= self._edge_probs.sum(axis=(1, 2), keepdims=True)

        # This represents information in the pairwise joint posterior minus
        # information in the individual factors.
        self._edge_trans = self._edge_probs.copy()
        for e, v1, v2 in tree.tree_grid.T:
            self._edge_trans[e, :, :] /= self._vert_probs[v1, np.newaxis, :]
            self._edge_trans[e, :, :] /= self._vert_probs[v2, :, np.newaxis]

        # This is the conditional distribution of features given latent.
        self._feat_cond = suffstats['feat_ss'].astype(np.float32) + feat_prior
        meas_probs = suffstats['meas_ss'].astype(np.float32) + meas_prior
        for v in range(V):
            beg, end = ragged_index[v:v + 2]
            self._feat_cond[beg:end, :] /= meas_probs[v, np.newaxis, :]

        # These are used to inspect and visualize latent structure.
        self._edge_logits = model['edge_logits']
        self._estimated_tree = tuple(
            estimate_tree(self._tree.complete_grid, self._edge_logits))
        self._tree.gc()

    @property
    def edge_logits(self):
        return self._edge_logits

    @property
    def estimate_tree(self):
        return self._estimated_tree

    def sample_tree(self, num_samples):
        """Returns a num_samples-long list of trees, each a list of pairs."""
        samples = []
        edge_logits = self.edge_logits
        edges = self.estimate_tree
        for _ in range(num_samples):
            edges = sample_tree(self._tree.complete_grid, edge_logits, edges)
            samples.append(edges)
        return samples

    @profile
    def sample(self, N, counts, data=None):
        """Draw N samples from the posterior distribution.

        Args:
          size: The number of samples to draw.
          counts: A [V]-shaped numpy array of requested counts of multinomials
            to sample.
          data: An optional single row of conditioning data, as a [R]-shaped
            ragged numpy array of multinomial counts,
            where R = server.ragged_size.

        Returns:
          An [N, R]-shaped numpy array of sampled multinomial data.
        """
        logger.debug('sampling data')
        V, E, M, R = self._VEMR
        if data is None:
            data = self._zero_row
        assert data.shape == (R, )
        assert data.dtype == np.int8
        assert counts.shape == (V, )
        assert counts.dtype == np.int8
        edge_trans = self._edge_trans
        feat_cond = self._feat_cond

        messages_in = self._vert_probs.copy()
        messages_out = np.tile(self._vert_probs[:, np.newaxis, :], (1, N, 1))
        vert_samples = np.zeros([V, N], np.int8)
        feat_samples = np.zeros([N, R], np.int8)
        range_N = np.arange(N, dtype=np.int32)

        for op, v, v2, e in self._program:
            if op == OP_UP:
                # Propagate upward from observed to latent.
                message = messages_in[v, :]
                beg, end = self._ragged_index[v:v + 2]
                for r in range(beg, end):
                    # This uses a with-replacement approximation that is exact
                    # for categorical data but approximate for multinomial.
                    if data[r]:
                        message *= feat_cond[r, :]**data[r]
            elif op == OP_IN:
                # Propagate latent state inward from children to v.
                message = messages_in[v, :]
                trans = edge_trans[e, :, :]
                if v > v2:
                    trans = trans.T
                message *= np.dot(trans, messages_in[v2, :])
                # Scale message for numerical stability.
                message /= message.max()
                message += SQRT_TINY
            elif op == OP_ROOT:
                # Process root node.
                messages_out[v, :, :] = messages_in[v, np.newaxis, :]
            elif op == OP_OUT:
                # Propagate latent state outward from parent to v.
                message = messages_out[v, :, :]
                message[...] = messages_in[v, np.newaxis, :]
                trans = edge_trans[e, :, :]
                if v2 > v:
                    trans = trans.T
                message *= trans[vert_samples[v2, :], :]
                # Scale message for numerical stability.
                message /= message.max(axis=1, keepdims=True)
                message += SQRT_TINY
            elif op == OP_DOWN:
                # Sample latent and observed assignment.
                message = messages_out[v, :, :]
                vert_samples[v, :] = sample_from_probs2(message)
                beg, end = self._ragged_index[v:v + 2]
                feat_block = feat_cond[beg:end, :].T
                probs = feat_block[vert_samples[v, :], :]
                samples_block = feat_samples[:, beg:end]
                for _ in range(counts[v]):
                    samples_block[range_N, sample_from_probs2(probs)] += 1

        return feat_samples

    @profile
    def logprob(self, data):
        """Compute non-normalized log probabilies of many rows of data.

        To compute conditional probabilty, use the identity:

          log P(data|evidence) = server.logprob(data + evidence)
                               - server.logprob(evidence)

        Args:
          data: A [N, R]-shaped ragged nummpy array of multinomial count data,
            where N is the number of rows, and R is server.ragged_size.

        Returns:
          An [N]-shaped numpy array of log probabilities.
        """
        logger.debug('computing logprob')
        V, E, M, R = self._VEMR
        N = data.shape[0]
        assert data.shape == (N, R)
        assert data.dtype == np.int8
        edge_trans = self._edge_trans
        feat_cond = self._feat_cond

        messages = np.tile(self._vert_probs[:, :, np.newaxis], (1, 1, N))
        assert messages.shape == (V, M, N)
        logprob = np.zeros(N, np.float32)

        for op, v, v2, e in self._program:
            message = messages[v, :, :]
            if op == OP_UP:
                # Propagate upward from observed to latent.
                beg, end = self._ragged_index[v:v + 2]
                for r in range(beg, end):
                    # This uses a with-replacement approximation that is exact
                    # for categorical data but approximate for multinomial.
                    power = data[np.newaxis, :, r]
                    message *= feat_cond[r, :, np.newaxis]**power
            elif op == OP_IN:
                # Propagate latent state inward from children to v.
                trans = edge_trans[e, :, :]
                if v > v2:
                    trans = trans.T
                message *= np.dot(trans, messages[v2, :, :])
                message_sum = message.sum(axis=0, keepdims=True)
                message /= message_sum
                logprob += np.log(message_sum[0, :])
            elif op == OP_ROOT:
                # Aggregate total log probability at the root node.
                logprob += np.log(message.sum(axis=0))
                return logprob

    @profile
    def marginals(self, data):
        """Compute observed marginals conditioned on data.

        Args:
          data: An [N, R]-shaped numpy array of row of conditioning data, as a
            ragged nummpy array of multinomial counts,
            where R = server.ragged_size.

        Returns:
          An real-valued array of the same shape as data.
        """
        logger.debug('computing marginals')
        V, E, M, R = self._VEMR
        N = data.shape[0]
        assert data.shape == (N, R)
        assert data.dtype == np.int8
        edge_trans = self._edge_trans
        feat_cond = self._feat_cond

        messages_in = np.empty([V, M, N], dtype=np.float32)
        messages_in[...] = self._vert_probs[:, :, np.newaxis]
        messages_out = np.empty_like(messages_in.copy())
        result = np.zeros([N, R], np.float32)

        for op, v, v2, e in self._program:
            message = messages_in[v, :, :]
            if op == OP_UP:
                # Propagate upward from observed to latent.
                beg, end = self._ragged_index[v:v + 2]
                for r in range(beg, end):
                    # This uses a with-replacement approximation that is exact
                    # for categorical data but approximate for multinomial.
                    power = data[np.newaxis, :, r]
                    message *= feat_cond[r, :, np.newaxis]**power
                messages_out[v, :, :] = message
            elif op == OP_IN:
                # Propagate latent state inward from children to v.
                trans = edge_trans[e, :, :]
                if v > v2:
                    trans = trans.T
                message *= np.dot(trans, messages_in[v2, :, :])
                # Scale message for numerical stability.
                message /= message.max(axis=0, keepdims=True)
                message += SQRT_TINY
            elif op == OP_OUT:
                # Propagate latent state outward from parent to v.
                trans = edge_trans[e, :, :]
                if v > v2:
                    trans = trans.T
                from_parent = np.dot(trans, messages_out[v2, :, :])
                messages_out[v, :, :] *= from_parent
                message *= from_parent
            elif op == OP_DOWN:
                # Propagate downward from latent state to observations.
                beg, end = self._ragged_index[v:v + 2]
                marginal = result[:, beg:end]
                marginal[...] = np.dot(feat_cond[beg:end, :], message).T
                marginal += TINY
                marginal /= marginal.sum(axis=1, keepdims=True)

        return result

    def observed_perplexity(self):
        """Compute perplexity = exp(entropy) of observed variables.

        Perplexity is an information theoretic measure of the number of
        clusters or latent classes. Perplexity is a real number in the range
        [1, M], where M is model_num_clusters.

        Returns:
          A [V]-shaped numpy array of perplexity.
        """
        V, E, M, R = self._VEMR
        observed_entropy = np.empty(V, dtype=np.float32)
        for v in range(V):
            beg, end = self._ragged_index[v:v + 2]
            observed_entropy[v] = entropy(
                np.dot(self._feat_cond[beg:end, :], self._vert_probs[v, :]))
        return np.exp(observed_entropy)

    def latent_perplexity(self):
        """Compute perplexity = exp(entropy) of latent variables.

        Perplexity is an information theoretic measure of the number of
        clusters or latent classes. Perplexity is a real number in the range
        [1, M], where M is model_num_clusters.

        Returns:
          A [V]-shaped numpy array of perplexity.
        """
        V, E, M, R = self._VEMR
        latent_entropy = np.array(
            [entropy(self._vert_probs[v, :]) for v in range(V)],
            dtype=np.float32)
        return np.exp(latent_entropy)

    @profile
    def latent_correlation(self):
        """Compute correlation matrix among latent features.

        This computes the generalization of Pearson's correlation to discrete
        data. Let I(X;Y) be the mutual information. Then define correlation as

          rho(X,Y) = sqrt(1 - exp(-2 I(X;Y)))

        Returns:
          A [V, V]-shaped numpy array of feature-feature correlations.
        """
        logger.debug('computing latent correlation')
        V, E, M, R = self._VEMR
        edge_probs = self._edge_probs
        vert_probs = self._vert_probs
        result = np.zeros([V, V], np.float32)
        for root in range(V):
            messages = np.empty([V, M, M])
            program = make_propagation_program(self._tree.tree_grid, root)
            for op, v, v2, e in program:
                if op == OP_ROOT:
                    # Initialize correlation at this node.
                    messages[v, :, :] = np.diagflat(vert_probs[v, :])
                elif op == OP_OUT:
                    # Propagate correlation outward from parent to v.
                    trans = edge_probs[e, :, :]
                    if v > v2:
                        trans = trans.T
                    messages[v, :, :] = np.dot(  #
                        trans / vert_probs[v2, np.newaxis, :],
                        messages[v2, :, :])
            for v in range(V):
                result[root, v] = correlation(messages[v, :, :])
        return result


class EnsembleServer(ServerBase):
    """Class for serving queries against a trained TreeCat ensemble."""

    def __init__(self, ensemble):
        logger.info('EnsembleServer of size %d', len(ensemble))
        assert ensemble
        ServerBase.__init__(self, ensemble[0]['suffstats']['ragged_index'])
        self._ensemble = [TreeCatServer(model) for model in ensemble]

        # These are used to inspect and visualize latent structure.
        self._edge_logits = self._ensemble[0].edge_logits.copy()
        for server in self._ensemble[1:]:
            self._edge_logits += server.edge_logits
        self._edge_logits /= len(self._ensemble)
        grid = self._ensemble[0]._tree.complete_grid
        self._estimated_tree = tuple(estimate_tree(grid, self._edge_logits))
        self._ensemble[0]._tree.gc()

    @property
    def edge_logits(self):
        return self._edge_logits

    @property
    def estimate_tree(self):
        return self._estimated_tree

    def sample_tree(self, num_samples):
        size = len(self._ensemble)
        pvals = np.ones(size, dtype=np.float32) / size
        sub_nums = np.random.multinomial(num_samples, pvals)
        samples = []
        for server, sub_num in zip(self._ensemble, sub_nums):
            samples += server.sample_tree(sub_num)
        np.random.shuffle(samples)
        assert len(samples) == num_samples
        return samples

    def sample(self, N, counts, data=None):
        size = len(self._ensemble)
        pvals = np.ones(size, dtype=np.float32) / size
        sub_Ns = np.random.multinomial(N, pvals)
        samples = np.concatenate([
            server.sample(sub_N, counts, data)
            for server, sub_N in zip(self._ensemble, sub_Ns)
        ])
        np.random.shuffle(samples)
        assert samples.shape[0] == N
        return samples

    def logprob(self, data):
        logprobs = np.stack(
            [server.logprob(data) for server in self._ensemble])
        logprobs = logsumexp(logprobs, axis=0)
        logprobs -= np.log(len(self._ensemble))
        assert logprobs.shape == (data.shape[0], )
        return logprobs

    def observed_perplexity(self):
        """Compute perplexity = exp(entropy) of observed variables.

        Perplexity is an information theoretic measure of the number of
        clusters or observed classes. Perplexity is a real number in the range
        [1, M], where M is model_num_clusters.

        Returns:
          A [V]-shaped numpy array of perplexity.
        """
        result = self._ensemble[0].observed_perplexity()
        for server in self._ensemble[1:]:
            result += server.observed_perplexity()
        result /= len(self._ensemble)
        return result

    def latent_perplexity(self):
        """Compute perplexity = exp(entropy) of latent variables.

        Perplexity is an information theoretic measure of the number of
        clusters or latent classes. Perplexity is a real number in the range
        [1, M], where M is model_num_clusters.

        Returns:
          A [V]-shaped numpy array of perplexity.
        """
        result = self._ensemble[0].latent_perplexity()
        for server in self._ensemble[1:]:
            result += server.latent_perplexity()
        result /= len(self._ensemble)
        return result

    def latent_correlation(self):
        """Compute correlation matrix among latent features.

        This computes the generalization of Pearson's correlation to discrete
        data. Let I(X;Y) be the mutual information. Then define correlation as

          rho(X,Y) = sqrt(1 - exp(-2 I(X;Y)))

        Returns:
          A [V, V]-shaped numpy array of feature-feature correlations.
        """
        result = self._ensemble[0].latent_correlation()
        for server in self._ensemble[1:]:
            result += server.latent_correlation()
        result /= len(self._ensemble)
        return result


class DataServer(object):
    """A schema-aware server interface for TreeCat and ensembles."""

    def __init__(self, dataset, ensemble):
        self._schema = dataset['schema']
        self._data = dataset['data']
        self._counts = guess_counts(self._schema['ragged_index'], self._data)
        if len(ensemble) == 1:
            self._server = TreeCatServer(ensemble[0])
        else:
            self._server = EnsembleServer(ensemble)
        self._feature_names = tuple(self._schema['feature_names'])

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def estimate_tree(self):
        """Returns a tuple of edges. Each edge is a (vertex,vertex) pair."""
        return self._server.estimate_tree

    def sample_tree(self, num_samples):
        """Returns a num_samples-long list of trees, each a list of pairs."""
        return self._server.sample_tree(num_samples)

    @property
    def edge_logits(self):
        return self._server.edge_logits

    def feature_density(self):
        """Returns a [V]-shaped array of feature densities in [0, 1]."""
        ragged_index = self._schema['ragged_index']
        V = len(ragged_index) - 1
        density = np.empty([V], np.float32)
        for v in range(V):
            beg, end = ragged_index[v:v + 2]
            density[v] = (self._data[:, beg:end].max(1) != 0).mean()
        return density

    def observed_perplexity(self):
        return self._server.observed_perplexity()

    def latent_perplexity(self):
        return self._server.latent_perplexity()

    def latent_correlation(self):
        return self._server.latent_correlation()

    def logprob(self, rows, evidence=None):
        data = import_rows(self._schema, rows)
        if evidence is None:
            return self._server.logprob(data)
        else:
            ragged_evidence = import_rows(self._schema, [evidence])
            return (self._server.logprob(data + ragged_evidence) -
                    self._server.logprob(data + evidence))

    def sample(self, N, evidence=None):
        if evidence is None:
            data = None
        else:
            data = import_rows(self._schema, [evidence])[0]
        ragged_samples = self._server.sample(N, self._counts, data)
        return export_rows(self._schema, ragged_samples)

    def median(self, evidence):
        ragged_evidence = import_rows(self._schema, evidence)
        data = self._server.median(self._counts, ragged_evidence)
        return export_rows(self._schema, data)

    def mode(self, evidence):
        ragged_evidence = import_rows(self._schema, evidence)
        data = self._server.mode(self, self._counts, ragged_evidence)
        return export_rows(self._schema, data)


def serve_model(dataset, model):
    """Create a server object from the given dataset and model.

    Args:
      dataset: Either a filename pointing to a dataset loadable by load_dataset
        or an already loaded dataset.
      model: Either the path to a TreeCat model or ensemble, or an already
        loaded model or ensemble.

    Returns:
      A DataServer object.
    """
    if isinstance(dataset, str):
        dataset = pickle_load(dataset)
    if isinstance(model, str):
        model = pickle_load(model)
    if isinstance(model, dict):
        model = [model]
    return DataServer(dataset, model)
