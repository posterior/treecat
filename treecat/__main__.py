from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

from parsable import parsable

from treecat.config import make_config
from treecat.format import guess_schema
from treecat.format import import_data
from treecat.format import pickle_dump
from treecat.format import pickle_load
from treecat.tables import TY_MULTINOMIAL
from treecat.tables import Table

parsable = parsable.Parsable()
parsable(guess_schema)
parsable(import_data)


@parsable
def train(dataset_in, ensemble_out, **options):
    """Train a TreeCat ensemble model on imported data."""
    from treecat.training import train_ensemble
    dataset = pickle_load(dataset_in)
    ragged_index = dataset['schema']['ragged_index']
    V = len(ragged_index) - 1
    feature_types = [TY_MULTINOMIAL] * V
    data = dataset['data']
    table = Table(feature_types, ragged_index, data)
    tree_prior = dataset['schema']['tree_prior']
    config = make_config(**options)
    ensemble = train_ensemble(table, tree_prior, config)
    pickle_dump(ensemble, ensemble_out)


if __name__ == '__main__':
    # This attempts to avoid deadlock when using to the default 'fork' method.
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('spawn')
    parsable()
