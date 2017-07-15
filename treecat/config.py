from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

DEFAULT_CONFIG = {
    'seed': 0,
    'model_num_clusters': 32,
    'model_ensemble_size': 8,
    'learning_init_epochs': 100,
    'learning_full_epochs': 1,
    'learning_estimate_tree': True,
    'learning_sample_tree_steps': 10,
    'serving_samples': 1024,
}


def make_config(**options):
    """Create a new global config dict with default values."""
    config = DEFAULT_CONFIG.copy()
    for key, value in options.items():
        if key not in config:
            raise ValueError('Unknown option: {}. Try one of:\n  {}'.format(
                key, '\n  '.join(sorted(config.keys()))))
        config[key] = int(value)
    return config
