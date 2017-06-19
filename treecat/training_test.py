from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import TreeCatTrainer
from treecat.training import train_model


def test_trainer_init():
    TreeCatTrainer(TINY_DATA, TINY_MASK, TINY_CONFIG)


def test_train_model():
    train_model(TINY_DATA, TINY_MASK, TINY_CONFIG)


# TODO Test get_annealing_schedule().
