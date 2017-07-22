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

parsable = parsable.Parsable()
parsable(guess_schema)
parsable(import_data)


@parsable
def train(dataset_in, ensemble_out, **options):
    """Train a TreeCat ensemble model on imported data."""
    from treecat.training import train_ensemble
    dataset = pickle_load(dataset_in)
    ragged_index = dataset['schema']['ragged_index']
    data = dataset['data']
    config = make_config(**options)
    ensemble = train_ensemble(ragged_index, data, config)
    pickle_dump(ensemble, ensemble_out)


if __name__ == '__main__':
    # This attempts to avoid deadlock when using to the default 'fork' method.
    multiprocessing.set_start_method('spawn')
    parsable()
