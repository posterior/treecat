from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import logging
import os
import shutil
import tempfile

LOG_LEVEL = int(os.environ.get('TREECAT_LOG_LEVEL', logging.CRITICAL))
LOG_FILENAME = os.environ.get('TREECAT_LOG_FILE')
LOG_FORMAT = '%(levelname).1s %(name)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, filename=LOG_FILENAME)

# https://oeis.org/A000272
NUM_SPANNING_TREES = [1, 1, 1, 3, 16, 125, 1296, 16807, 262144, 4782969]


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))


@contextlib.contextmanager
def tempdir():
    dirname = tempfile.mkdtemp()
    try:
        yield dirname
    finally:
        shutil.rmtree(dirname)
