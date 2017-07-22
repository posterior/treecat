from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from treecat import validate
from treecat.generate import generate_dataset_file
from treecat.testutil import TESTDATA
from treecat.testutil import tempdir

PARAM_CSV = os.path.join(TESTDATA, 'tuning.csv')


def test_train_eval():
    dataset_path = generate_dataset_file(5, 7)
    with tempdir() as dirname:
        models_dir = os.path.join(dirname, 'models')
        result_path = os.path.join(dirname, 'tune_clusters.pkz')
        validate.train(dataset_path, PARAM_CSV, models_dir)
        validate.eval(dataset_path, PARAM_CSV, models_dir, result_path)
        assert os.path.exists(result_path)
