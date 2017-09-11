from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from subprocess import check_call

import numpy as np

from treecat.testutil import TESTDATA
from treecat.testutil import in_tempdir


def test_quickstart():
    with in_tempdir():
        # 1. Format your data.
        shutil.copy(os.path.join(TESTDATA, 'tiny_data.csv'), 'data.csv')

        # 2. Generate two schema files.
        check_call(
            'treecat guess-schema data.csv types.csv values.csv', shell=True)

        # 3. Import csv files.
        check_call(
            "treecat import-data data.csv types.csv values.csv '' dataset.pkz",
            shell=True)

        # 4. Train an ensemble model.
        check_call('treecat train dataset.pkz model.pkz', shell=True)

        # 5. Load your trained model.
        from treecat.serving import serve_model
        server = serve_model('dataset.pkz', 'model.pkz')

        # 6. Run queries.
        samples = server.sample(100, evidence={'genre': 'drama'})
        print(np.mean([s['rating'] for s in samples]))
        # Explore feature structure.
        print(server.latent_correlation())
