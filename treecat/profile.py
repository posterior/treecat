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
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.testutil import tempdir

PYTHON = sys.executable
FILE = os.path.abspath(__file__)

parsable = parsable.Parsable()


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
def train_files(dataset_path, config_path):
    """INTERNAL Train from pickled dataset, config."""
    from treecat.training import train_model
    dataset = pickle_load(dataset_path)
    config = pickle_load(config_path)
    train_model(dataset['ragged_index'], dataset['data'], config['config'])


@parsable
def serve_files(model_path, config_path):
    """INTERNAL Serve from pickled model, config."""
    from treecat.serving import serve_model
    import numpy as np
    model = pickle_load(model_path)
    config = pickle_load(config_path)
    server = serve_model(model['tree'], model['suffstats'], config['config'])
    counts = np.ones(model['tree'].num_vertices, np.int8)
    for _ in range(1000):
        sample = server.sample(counts)
        server.logprob(sample)


@parsable
def train(rows=100, cols=10, epochs=5, tool='timers'):
    """Profile TreeCatTrainer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_dataset_file
    config = DEFAULT_CONFIG.copy()
    config['learning_annealing_epochs'] = epochs
    dataset_path = generate_dataset_file(rows, cols)
    with tempdir() as dirname:
        config_path = os.path.join(dirname, 'config.pkl.gz')
        pickle_dump({'config': config}, config_path)
        cmd = [FILE, 'train_files', dataset_path, config_path]
        run_with_tool(cmd, tool, dirname)


@parsable
def serve(rows=100, cols=10, cats=4, tool='timers'):
    """Profile TreeCatServer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_model_file
    config = DEFAULT_CONFIG.copy()
    model_path = generate_model_file(rows, cols, cats)
    with tempdir() as dirname:
        config_path = os.path.join(dirname, 'config.pkl.gz')
        pickle_dump({'config': config}, config_path)
        cmd = [FILE, 'serve_files', model_path, config_path]
        run_with_tool(cmd, tool, dirname)


if __name__ == '__main__':
    parsable()
