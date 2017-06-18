from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import logging
import os
from collections import Counter
from collections import defaultdict
from timeit import default_timer

DEBUG_LEVEL = int(os.environ.get('TREECAT_DEBUG_LEVEL', 0))
LOG_LEVEL = int(os.environ.get('TREECAT_LOG_LEVEL', logging.CRITICAL))
PROFILING = (LOG_LEVEL <= 15)
LOG_FILENAME = os.environ.get('TREECAT_LOG_FILE')
LOG_FORMAT = '%(levelname).1s %(name)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, filename=LOG_FILENAME)
logger = logging.getLogger(__name__)


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))


def sizeof(array):
    '''Computes byte size of numpy.ndarray or tensorflow.Tensors.
    Args:
      array: A numpy array or tensorflow Tensor.

    Returns:
      Memory footprint in bytes.
    '''
    dtype = array.dtype
    size = dtype.size if hasattr(dtype, 'size') else dtype.itemsize
    for dim in array.shape:
        size *= int(dim)
    return size


class ProfilingSet(defaultdict):
    __getattr__ = defaultdict.__getitem__
    __setattr__ = defaultdict.__setitem__


class ProfileTimer(object):
    __slots__ = ['elapsed', 'count']

    def __init__(self):
        self.elapsed = 0.0
        self.count = 0

    def __enter__(self):
        self.elapsed -= default_timer()

    def __exit__(self, type, value, traceback):
        self.elapsed += default_timer()
        self.count += 1


def profile_timed(fun):
    '''Decorator for time-based profiling of individual functions.'''
    if not PROFILING:
        return fun
    timer = TIMERS[fun]

    @functools.wraps(fun)
    def profiled_fun(*args, **kwargs):
        with timer:
            return fun(*args, **kwargs)

    return profiled_fun


# Allow line_profiler to override profile_timed by adding it to __builtins__.
profile = __builtins__.get('profile', profile_timed)

# Use these to access global profiling state.
TIMERS = defaultdict(ProfileTimer)
COUNTERS = ProfilingSet(lambda: 0)
HISTOGRAMS = ProfilingSet(Counter)


def log_profile_counters():
    logger.info('-' * 64)
    logger.info('Counters:')
    for name, histogram in sorted(HISTOGRAMS.items()):
        logger.info('{: >10s} {}'.format('Count', name))
        for value, count in sorted(histogram.items()):
            logger.info('{: >10d} {}'.format(count, value))
    logger.info('{: >10s} {}'.format('Count', 'Counter'))
    for name, count in sorted(COUNTERS.items()):
        logger.info('{: >10d} {}'.format(count, name))


def log_profile_timers():
    times = [(t.elapsed, t.count, f) for (f, t) in TIMERS.items()]
    times.sort(reverse=True, key=lambda x: x[0])
    logger.info('-' * 64)
    logger.info('Timers:')
    logger.info('{: >10} {: >10} {}'.format('Seconds', 'Calls', 'Function'))
    for time, count, fun in times:
        logger.info('{: >10.3f} {: >10} {}.{}'.format(
            time, count, fun.__module__, fun.__name__))


if PROFILING:
    atexit.register(log_profile_timers)
    atexit.register(log_profile_counters)
