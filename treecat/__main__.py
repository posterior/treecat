from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import sys
from subprocess import Popen
from subprocess import check_call

from parsable import parsable
from treecat.config import DEFAULT_CONFIG
from treecat.persist import pickle_dump
from treecat.persist import pickle_load
from treecat.testutil import tempdir

PYTHON = sys.executable


def run_with_tool(cmd, tool, dirname):
    profile_path = os.path.join(dirname, 'profile_train.prof')
    if tool == 'timers':
        env = os.environ.copy()
        env.setdefault('TREECAT_LOG_LEVEL', '15')
        Popen([PYTHON, '-O'] + cmd, env=env).wait()
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
def train(dataset_path, config_path):
    """Train from pickled data, mask, config."""
    from treecat.training import train_model
    data, mask = pickle_load(dataset_path)
    config = pickle_load(config_path)
    train_model(data, mask, config)


@parsable
def profile_train(rows=100, cols=10, epochs=5, tool='timers', engine='numpy'):
    """Profile TreeCatTrainer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_dataset
    config = DEFAULT_CONFIG.copy()
    config['learning_annealing_epochs'] = epochs
    config['engine'] = engine
    cats = config['model_num_categories']
    dataset_path = generate_dataset(rows, cols, cats)
    with tempdir() as dirname:
        config_path = os.path.join(dirname, 'config.pkl.gz')
        pickle_dump(config, config_path)
        cmd = [os.path.abspath(__file__), 'train', dataset_path, config_path]
        run_with_tool(cmd, tool, dirname)


@parsable
def serve(model_path, config_path):
    """Train from pickled data, mask, config."""
    from treecat.serving import serve_model
    model = pickle_load(model_path)
    config = pickle_load(config_path)
    server = serve_model(model['tree'], model['suffstats'], config)
    server.correlation()


@parsable
def profile_serve(rows=100, cols=10, tool='timers', engine='numpy'):
    """Profile TreeCatServer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_model
    config = DEFAULT_CONFIG.copy()
    cats = config['model_num_categories']
    model_path = generate_model(rows, cols, cats)
    config['engine'] = engine
    with tempdir() as dirname:
        config_path = os.path.join(dirname, 'config.pkl.gz')
        pickle_dump(config, config_path)
        cmd = [os.path.abspath(__file__), 'serve', model_path, config_path]
        run_with_tool(cmd, tool, dirname)


if __name__ == '__main__':
    parsable()
