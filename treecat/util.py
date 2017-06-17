from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

LOG_LEVEL = int(os.environ.get('TREECAT_LOG_LEVEL', logging.CRITICAL))
LOG_FILENAME = os.environ.get('TREECAT_LOG_FILE')
LOG_FORMAT = '%(levelname).1s %(name)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, filename=LOG_FILENAME)


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))
