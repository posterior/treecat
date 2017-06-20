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
def train(filename):
    """Train from a pickled (data, mask, config) tuple."""
    from treecat.training import train_model
    data, mask, config = pickle_load(filename)
    train_model(data, mask, config)


@parsable
def profile_train(rows=100, cols=10, cats=4, epochs=5, tool='timers'):
    """Profile TreeCatTrainer.train() on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler
    """
    from treecat.config import DEFAULT_CONFIG
    from treecat.generate import generate_dataset
    config = deepcopy(DEFAULT_CONFIG)
    config['num_categories'] = cats
    config['annealing']['epochs'] = epochs
    data, mask = generate_dataset(rows, cols, config=config)
    task = (data, mask, config)
    with tempdir() as dirname:
        task_path = os.path.join(dirname, 'profile_train.pkl.gz')
        profile_path = os.path.join(dirname, 'profile_train.prof')
        pickle_dump(task, task_path)
        cmd = [os.path.abspath(__file__), 'train', task_path]
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
            env.setdefault('TREECAT_LOG_LEVEL', '15')
            Popen([PYTHON, '-O'] + cmd, env=env).wait()

        elif tool == 'line_profiler':
            check_call(['kernprof', '-l', '-v', '-o', profile_path] + cmd)
        else:
            raise ValueError('Unknown tool: {}'.format(tool))


# TODO Support tensorflow profiling.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/tfprof

if __name__ == '__main__':
    parsable()
