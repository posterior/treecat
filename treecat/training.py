from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging

import numpy as np

from treecat.structure import TreeStructure
from treecat.util import COUNTERS
from treecat.util import art_logger
from treecat.util import sizeof

logger = logging.getLogger(__name__)


class TrainerBase(object):
    """Base class for training a TreeCat model."""

    def __init__(self, data, mask, config):
        """Initialize a model in an unassigned state.

        Args:
            data: A 2D array of categorical data.
            mask: A 2D array of presence/absence, where present = True.
            config: A global config dict.
        """
        data = np.asarray(data, np.int32)
        mask = np.asarray(mask, np.bool_)
        num_rows, num_features = data.shape
        assert data.shape == mask.shape
        self._data = data
        self._mask = mask
        self._config = config
        self._seed = config['seed']
        self._assigned_rows = set()
        self.assignments = np.zeros(data.shape, dtype=np.int32)
        self.suffstats = {}
        self.tree = TreeStructure(num_features)

        COUNTERS.footprint_training_data = sizeof(self._data)
        COUNTERS.footprint_training_mask = sizeof(self._mask)
        COUNTERS.footprint_training_assignments = sizeof(self.assignments)

    def add_row(self, row_id):
        raise NotImplementedError()

    def remove_row(self, row_id):
        raise NotImplementedError()

    def sample_tree(self):
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()

    def train(self):
        """Train a TreeCat model using subsample-annealed MCMC.

        Let N be the number of data rows and V be the number of features.

        Returns:
          A trained model as a dictionary with keys:
            tree: A TreeStructure instance with the learned latent structure.
            suffstats: Sufficient statistics of features, vertices, and
              edges.
            assignments: An [N, V] numpy array of latent cluster ids for each
              cell in the dataset.
        """
        logger.info('train()')
        num_rows = self._data.shape[0]
        for action, row_id in get_annealing_schedule(num_rows, self._config):
            if action == 'add_row':
                art_logger('+')
                self.add_row(row_id)
            elif action == 'remove_row':
                art_logger('-')
                self.remove_row(row_id)
            else:
                art_logger('\n')
                self.sample_tree()
        self.finish()
        return {
            'config': self._config,
            'tree': self.tree,
            'suffstats': self.suffstats,
            'assignments': self.assignments,
        }


def train_model(data, mask, config):
    """Train a TreeCat model using subsample-annealed MCMC.

    Let N be the number of data rows and V be the number of features.

    Returns:
      A trained model as a dictionary with keys:
        tree: A TreeStructure instance with the learned latent structure.
        suffstats: Sufficient statistics of features, vertices, and
          edges.
        assignments: An [N, V] numpy array of latent cluster ids for each
          cell in the dataset.
    """
    if config['engine'] == 'numpy':
        from treecat.np_engine import NumpyTrainer as Trainer
    elif config['engine'] == 'tensorflow':
        from treecat.tf_engine import TensorflowTrainer as Trainer
    else:
        raise ValueError('Unknown engine: {}'.format(config['engine']))
    return Trainer(data, mask, config).train()


def get_annealing_schedule(num_rows, config):
    """Iterator for subsample annealing yielding (action, arg) pairs.

    Actions are one of: 'add_row', 'remove_row', or 'batch'.
    The add and remove actions each provide a row_id arg.
    """
    # Randomly shuffle rows.
    row_ids = list(range(num_rows))
    np.random.seed(config['seed'])
    np.random.shuffle(row_ids)
    row_to_add = itertools.cycle(row_ids)
    row_to_remove = itertools.cycle(row_ids)

    # Use a linear annealing schedule.
    epochs = float(config['annealing_epochs'])
    add_rate = epochs
    remove_rate = epochs - 1.0
    state = epochs * config['annealing_init_rows']

    # Perform batch operations between batches.
    num_fresh = 0
    num_stale = 0
    while num_fresh + num_stale != num_rows:
        if state >= 0.0:
            yield 'add_row', next(row_to_add)
            state -= remove_rate
            num_fresh += 1
        else:
            yield 'remove_row', next(row_to_remove)
            state += add_rate
            num_stale -= 1
        if num_stale == 0 and num_fresh > 0:
            yield 'batch', None
            num_stale = num_fresh
            num_fresh = 0
