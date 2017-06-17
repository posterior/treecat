from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os
import subprocess
from copy import deepcopy

from parsable import parsable

from treecat.testutil import tempdir


@parsable
def fit(model_in, model_out=None):
    '''Fit a pickled model and optionally save it.'''
    from treecat.engine import Model
    assert Model  # Pacify linter.
    with open(model_in, 'rb') as f:
        model = pickle.load(f)
    print('Fitting model')
    model.fit()
    print('Done fitting model')
    if model_out is not None:
        with open(model_out, 'wb') as f:
            pickle.dump(model, f)


@parsable
def profile_fit_py(rows=100, cols=10, cats=4, density=0.9, epochs=5):
    '''Profile python part of Model.fit() on a random dataset.'''
    from treecat.engine import DEFAULT_CONFIG
    from treecat.engine import Model
    from treecat.generate import generate_dataset
    config = deepcopy(DEFAULT_CONFIG)
    config['num_categories'] = cats
    config['annealing']['epochs'] = epochs
    data, mask = generate_dataset(rows, cols, density, config)
    model = Model(data, mask, config)
    with tempdir() as dirname:
        model_path = os.path.join(dirname, 'profile_fit.model.pkl')
        profile_path = os.path.join(dirname, 'profile_fit.prof')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        subprocess.check_call([
            'python',
            '-m',
            'cProfile',
            '-o',
            profile_path,
            os.path.abspath(__file__),
            'fit',
            model_path,
        ])
        subprocess.check_call(['snakeviz', profile_path])


# TODO Add a profile_fit_tf c
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/tfprof

if __name__ == '__main__':
    parsable()
