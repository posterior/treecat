from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import sys
from subprocess import CalledProcessError
from subprocess import Popen
from subprocess import check_call

from parsable import parsable

from treecat.config import DEFAULT_CONFIG
from treecat.persist import pickle_dump
from treecat.persist import pickle_load
from treecat.testutil import tempdir

PYTHON = sys.executable
FILE = os.path.abspath(__file__)


def run_with_tool(cmd, tool, dirname):
    profile_path = os.path.join(dirname, 'profile_train.prof')
    if tool == 'timers':
        env = os.environ.copy()
        env.setdefault('TREECAT_LOG_LEVEL', '15')
        cmd = [PYTHON, '-O'] + cmd
        ret = Popen(cmd, env=env).wait()
        if ret:
            raise CalledProcessError(returncode=ret, cmd=cmd)
    elif tool == 'time':
        if platform.platform().startswith('Darwin'):
            gnu_time = 'gtime'
        else:
            gnu_time = '/usr/bin/time'
        check_call([gnu_time, '-v', PYTHON, '-O'] + cmd)
    elif tool == 'snakeviz':
        check_call([PYTHON, '-m', 'cProfile', '-o', profile_path] + cmd)
        check_call(['snakeviz', profile_path])
    elif tool == 'line_profiler':
        check_call(['kernprof', '-l', '-v', '-o', profile_path] + cmd)
    elif tool == 'pdb':
        check_call([PYTHON, '-m', 'pdb'] + cmd)
    else:
        raise ValueError('Unknown tool: {}'.format(tool))


@parsable
def train_files(data_path, config_path):
    """INTERNAL Train from pickled data, config."""
    from treecat.training import train_model
    data = pickle_load(data_path)
    config = pickle_load(config_path)
    train_model(data, config)


@parsable
def train(rows=100, cols=10, epochs=5, tool='timers'):
    """Profile TreeCatTrainer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_dataset_file
    config = DEFAULT_CONFIG.copy()
    config['learning_annealing_epochs'] = epochs
    data_path = generate_dataset_file(rows, cols)
    with tempdir() as dirname:
        config_path = os.path.join(dirname, 'config.pkl.gz')
        pickle_dump(config, config_path)
        cmd = [FILE, 'train_files', data_path, config_path]
        run_with_tool(cmd, tool, dirname)


if __name__ == '__main__':
    parsable()
