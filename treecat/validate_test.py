from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from treecat.generate import generate_dataset_file
from treecat.testutil import tempdir
from treecat.validate import tune_epochs


def test_tune_epochs():
    dataset_path = generate_dataset_file(5, 7)
    with tempdir() as dirname:
        result_path = os.path.join(dirname, 'tune_epochs.pkz')
        tune_epochs(
            dataset_path,
            result_path,
            2,
            3,
            4,
            5,
            6,
            model_ensemble_size=3,
            model_num_clusters=5)
