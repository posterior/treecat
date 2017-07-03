from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import logging
import os
import sys
from collections import Counter
from collections import defaultdict
from timeit import default_timer

import numpy as np

TREECAT_JIT = int(os.environ.get('TREECAT_JIT', 1))
DEBUG_LEVEL = int(os.environ.get('TREECAT_DEBUG_LEVEL', 0))
LOG_LEVEL = int(os.environ.get('TREECAT_LOG_LEVEL', logging.CRITICAL))
LOG_ART = int(os.environ.get('TREECAT_LOG_ART', 0))
PROFILING = (LOG_LEVEL <= 15)
LOG_FILENAME = os.environ.get('TREECAT_LOG_FILE')
LOG_FORMAT = '%(levelname).1s %(name)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, filename=LOG_FILENAME)
logger = logging.getLogger(__name__)


def no_jit(*args, **kwargs):
    if not kwargs and len(args) == 1 and callable(args[0]):
        return args[0]
    return no_jit


if TREECAT_JIT:
    try:
        from numba import jit
    except ImportError:
        jit = no_jit
else:
    jit = no_jit


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))


def log_art(art):
    sys.stderr.write(art)
    sys.stderr.flush()


art_logger = log_art if LOG_ART else (lambda art: None)


def sizeof(array):
    """Computes byte size of numpy.ndarray.
    Args:
      array: A numpy ndarray.

    Returns:
      Memory footprint in bytes.
    """
    dtype = array.dtype
    size = dtype.size if hasattr(dtype, 'size') else dtype.itemsize
    for dim in array.shape:
        size *= int(dim)
    return size


def sample_from_probs(probs):
    """Equivalent to np.random.choice(len(probs), p=probs)."""
    # Note: np.random.multinomial is faster than np.random.choice,
    # but np.random.multinomial is pickier about non-normalized probs.
    try:
        return np.random.multinomial(1, probs).argmax()
    except ValueError:
        COUNTERS.np_random_multinomial_value_error += 1
        return probs.argmax()


def sample_from_probs2(probs, out=None):
    """Vectorized sampler from categorical distribution."""
    # Adapted from https://stackoverflow.com/questions/40474436
    assert len(probs.shape) == 2
    u = np.random.rand(probs.shape[0], 1)
    cdf = probs.cumsum(axis=1)
    return (u < cdf).argmax(axis=1, out=out)


def make_ragged_index(columns):
    """Make an index to hold data in a ragged array.

    Args:
      columns: A list of numpy arrays of varying size.

    Returns:
      A [len(columns) + 1]-shaped array of begin,end positions of each column.
    """
    ragged_index = np.zeros([len(columns) + 1], dtype=np.int32)
    ragged_index[0] = 0
    for v, column in enumerate(columns):
        ragged_index[v + 1] = ragged_index[v] + column.shape[-1]
    return ragged_index


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
    """Decorator for time-based profiling of individual functions."""
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
