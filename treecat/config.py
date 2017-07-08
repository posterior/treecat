from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

DEFAULT_CONFIG = {
    'seed': 0,
    'model_num_clusters': 32,
    'model_ensemble_size': 8,
    'learning_init_rows': 2,
    'learning_epochs': 100.0,
    'learning_sample_tree_steps': 10,
    'learning_estimate_tree_steps': 1,
    'serving_samples': 1024,
}


def make_default_config():
    """Create a new global config dict with default values."""
    return DEFAULT_CONFIG.copy()
