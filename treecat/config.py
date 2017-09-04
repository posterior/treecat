from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Changes to keys invalidates configs serialized by serialize_config.
DEFAULT_CONFIG = {
    'seed': 0,
    'model_num_clusters': 32,
    'model_ensemble_size': 8,
    'learning_init_epochs': 100,
    'learning_full_epochs': 1,
    'learning_estimate_tree': True,
    'learning_sample_tree_rate': 3,
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


def serialize_config(config):
    """Serialize a config dict to a short string for use in filenames."""
    keys = sorted(config.keys())
    assert keys == sorted(make_config().keys())
    return '-'.join(str(int(config[key])) for key in keys)


def deserialize_config(config_str):
    """Deserialize a config dict form a short string."""
    config = make_config()
    keys = sorted(config.keys())
    values = config_str.split('-')
    assert len(keys) == len(values)
    for key, value_str in zip(keys, values):
        config[key] = int(value_str)
    return config
