from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import sys
from copy import deepcopy
from subprocess import Popen
from subprocess import check_call

from parsable import parsable
from treecat.persist import pickle_dump
from treecat.persist import pickle_load
from treecat.testutil import tempdir

PYTHON = sys.executable


@parsable
def fit(model_in, model_out=None):
    '''Fit a pickled model and optionally save it.'''
    from treecat.training import Model
    assert Model  # Pacify linter.
    model = pickle_load(model_in)
    print('Fitting model')
    model.fit()
    print('Done fitting model')
    if model_out is not None:
        pickle_dump(model, model_out)


@parsable
def profile_fit(rows=100, cols=10, cats=4, epochs=5, tool='timers'):
    '''Profile Model.fit() on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler
    '''
    from treecat.training import DEFAULT_CONFIG
    from treecat.training import Model
    from treecat.generate import generate_dataset
    config = deepcopy(DEFAULT_CONFIG)
    config['num_categories'] = cats
    config['annealing']['epochs'] = epochs
    data, mask = generate_dataset(rows, cols, config=config)
    model = Model(data, mask, config)
    with tempdir() as dirname:
        model_path = os.path.join(dirname, 'profile_fit.model.pkl.gz')
        profile_path = os.path.join(dirname, 'profile_fit.prof')
        pickle_dump(model, model_path)
        cmd = [os.path.abspath(__file__), 'fit', model_path]
        if tool == 'time':
            if platform.platform().startswith('Darwin'):
                gnu_time = 'gtime'
            else:
                gnu_time = '/usr/bin/time'
            check_call([gnu_time, '-v', PYTHON, '-O'] + cmd)
        elif tool == 'snakeviz':
            check_call([PYTHON, '-m', 'cProfile', '-o', profile_path] + cmd)
            check_call(['snakeviz', profile_path])
        elif tool == 'timers':
            env = os.environ.copy()
            env['TREECAT_PROFILE_TIME'] = '1'
            Popen([PYTHON, '-O'] + cmd, env=env).wait()

        elif tool == 'line_profiler':
            check_call(['kernprof', '-l', '-v', '-o', profile_path] + cmd)
        else:
            raise ValueError('Unknown tool: {}'.format(tool))


# TODO Add a profile_fit_tf c
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/tfprof

if __name__ == '__main__':
    parsable()
