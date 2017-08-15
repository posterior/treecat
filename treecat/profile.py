from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import sys
from subprocess import CalledProcessError
from subprocess import Popen
from subprocess import check_call

import numpy as np
from parsable import parsable

from treecat.config import make_config
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.testutil import tempdir

PYTHON = sys.executable
FILE = os.path.abspath(__file__)

parsable = parsable.Parsable()


def check_call_env(cmd, env):
    ret = Popen(cmd, env=env).wait()
    if ret:
        raise CalledProcessError(returncode=ret, cmd=cmd)


def run_with_tool(cmd, tool, dirname):
    profile_path = os.path.join(dirname, 'profile_train.prof')
    env = os.environ.copy()
    env['TREECAT_THREADS'] = '1'
    if tool == 'timers':
        env.setdefault('TREECAT_PROFILE', '1')
        env.setdefault('TREECAT_LOG_LEVEL', '20')
        check_call_env([PYTHON, '-O'] + cmd, env)
    elif tool == 'time':
        if platform.platform().startswith('Darwin'):
            gnu_time = 'gtime'
        else:
            gnu_time = '/usr/bin/time'
        check_call_env([gnu_time, '-v', PYTHON, '-O'] + cmd, env)
    elif tool == 'snakeviz':
        check_call_env([PYTHON, '-m', 'cProfile', '-o', profile_path] + cmd,
                       env)
        check_call(['snakeviz', profile_path])
    elif tool == 'line_profiler':
        check_call_env(['kernprof', '-l', '-v', '-o', profile_path] + cmd, env)
    elif tool == 'pdb':
        check_call_env([PYTHON, '-m', 'pdb'] + cmd, env)
    else:
        raise ValueError('Unknown tool: {}'.format(tool))


@parsable
def train_files(dataset_path, config_path):
    """INTERNAL Train from pickled dataset, config."""
    from treecat.training import train_ensemble
    dataset = pickle_load(dataset_path)
    ragged_index = dataset['schema']['ragged_index']
    V = ragged_index.shape[0] - 1
    K = V * (V - 1) // 2
    tree_prior = np.zeros(K, dtype=np.float32)
    config = pickle_load(config_path)
    train_ensemble(ragged_index, dataset['data'], tree_prior, config)


@parsable
def serve_files(model_path, config_path, num_samples):
    """INTERNAL Serve from pickled model, config."""
    from treecat.serving import TreeCatServer
    import numpy as np
    model = pickle_load(model_path)
    config = pickle_load(config_path)
    model['config'] = config
    server = TreeCatServer(model)
    counts = np.ones(model['tree'].num_vertices, np.int8)
    samples = server.sample(int(num_samples), counts)
    server.logprob(samples)
    server.median(counts, samples)
    server.latent_correlation()


@parsable
def train(rows=100, cols=10, epochs=5, clusters=32, tool='timers'):
    """Profile TreeCatTrainer on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_dataset_file
    config = make_config(
        learning_init_epochs=epochs,
        model_num_clusters=clusters,
        model_ensemble_size=1, )
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
        cmd = [FILE, 'serve_files', model_path, config_path, str(rows)]
        run_with_tool(cmd, tool, dirname)


@parsable
def eval(rows=100, cols=10, cats=4, tool='timers'):
    """Profile treecat.validate.eval on a random dataset.
    Available tools: timers, time, snakeviz, line_profiler, pdb
    """
    from treecat.generate import generate_dataset_file
    from treecat.validate import train
    dataset_path = generate_dataset_file(rows, cols)
    validate_py = os.path.join(os.path.dirname(FILE), 'validate.py')
    with tempdir() as dirname:
        param_csv_path = os.path.join(dirname, 'param.csv')
        with open(param_csv_path, 'w') as f:
            f.write('learning_init_epochs\n2')
        train(dataset_path, param_csv_path, dirname, learning_init_epochs=2)
        cmd = [
            validate_py,
            'eval',
            dataset_path,
            param_csv_path,
            dirname,
            os.path.join(dirname, 'tuning.pkz'),
            'learning_init_epochs=2',
        ]
        run_with_tool(cmd, tool, dirname)


if __name__ == '__main__':
    parsable()
