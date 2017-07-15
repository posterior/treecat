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

from treecat.config import make_config
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
    from treecat.training import train_ensemble
    dataset = pickle_load(dataset_path)
    config = pickle_load(config_path)
    train_ensemble(dataset['schema']['ragged_index'], dataset['data'], config)


@parsable
def serve_files(model_path, config_path):
    """INTERNAL Serve from pickled model, config."""
    from treecat.serving import TreeCatServer
    import numpy as np
    model = pickle_load(model_path)
    config = pickle_load(config_path)
    model['config'] = config
    server = TreeCatServer(model)
    num_samples = config['serving_samples']
    counts = np.ones(model['tree'].num_vertices, np.int8)
    samples = server.sample(num_samples, counts)
    server.logprob(samples)
    server.latent_correlation()


@parsable
def train(rows=100, cols=10, epochs=5, ensemble=1, tool='timers'):
    """Profile TreeCatTrainer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_dataset_file
    config = make_config(
        learning_init_epochs=epochs, model_ensemble_size=ensemble)
    dataset_path = generate_dataset_file(rows, cols)
    with tempdir() as dirname:
        config_path = os.path.join(dirname, 'config.pkz')
        pickle_dump(config, config_path)
        cmd = [FILE, 'train_files', dataset_path, config_path]
        run_with_tool(cmd, tool, dirname)


@parsable
def serve(rows=100, cols=10, cats=4, tool='timers'):
    """Profile TreeCatServer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_model_file
    config = make_config()
    model_path = generate_model_file(rows, cols, cats)
    with tempdir() as dirname:
        config_path = os.path.join(dirname, 'config.pkz')
        pickle_dump(config, config_path)
        cmd = [FILE, 'serve_files', model_path, config_path]
        run_with_tool(cmd, tool, dirname)


if __name__ == '__main__':
    parsable()
