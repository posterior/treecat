from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from treecat.__main__ import train
from treecat.generate import generate_dataset_file
from treecat.testutil import tempdir


def test_train():
    dataset_path = generate_dataset_file(10, 10)
    with tempdir() as dirname:
        ensemble_path = os.path.join(dirname, 'ensemble.pkz')
        train(
            dataset_path,
            ensemble_path,
            model_ensemble_size='3',
            learning_init_epochs='3')
        assert os.path.exists(ensemble_path)
