from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

DEFAULT_CONFIG = {
    'seed': 0,
    'num_categories': 3,  # E.g. CSE-IT data.
    'num_clusters': 32,
    'sample_tree_steps': 32,
    'annealing': {
        'init_rows': 2,
        'epochs': 100.0,
    },
}
