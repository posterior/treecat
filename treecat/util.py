from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import contextlib
import functools
import logging
import multiprocessing
import os
from collections import Counter
from collections import defaultdict
from timeit import default_timer

import numpy as np

from six.moves import map
from six.moves import range

TREECAT_JIT = int(os.environ.get('TREECAT_JIT', 1))
LOG_LEVEL = int(os.environ.get('TREECAT_LOG_LEVEL', logging.CRITICAL))
PROFILING = (LOG_LEVEL <= 15)
LOG_FILENAME = os.environ.get('TREECAT_LOG_FILE')
LOG_FORMAT = '%(levelname).1s %(process)d %(name)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, filename=LOG_FILENAME)
logger = logging.getLogger(__name__)

TINY = np.finfo(np.float32).tiny
SQRT_TINY = np.float32(np.finfo(np.float32).tiny**0.5)


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


@jit
def jit_random_seed(seed):
    np.random.seed(seed)


def set_random_seed(seed):
    """Set random seeds for both numpy and numba."""
    np.random.seed(seed)
    jit_random_seed(seed)


@contextlib.contextmanager
def np_printoptions(**kwargs):
    """Context manager to temporarily set numpy print options."""
    old = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield
    np.set_printoptions(**old)


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


@jit(nopython=True, cache=True)
def jit_sample_from_probs(probs):
    """Sample from a vector of non-normalized probabilitites.

    Args:
      probs: An [M]-shaped numpy array of non-normalized probabilities.

    Returns:
      An integer in range(M).
    """
    cdf = probs.cumsum()
    return (np.random.rand() * cdf[-1] < cdf).argmax()


def sample_from_probs2(probs, out=None):
    """Sample from multiple vectors of non-normalized probabilities.

    Args:
      probs: An [N, M]-shaped numpy array of non-normalized probabilities.
      out: An optional destination for the result.

    Returns:
      An [N]-shaped numpy array of integers in range(M).
    """
    # Adapted from https://stackoverflow.com/questions/40474436
    assert len(probs.shape) == 2
    cdf = probs.cumsum(axis=1)
    u = np.random.rand(probs.shape[0], 1) * cdf[:, -1, np.newaxis]
    return (u < cdf).argmax(axis=1, out=out)


def quantize_from_probs2(probs, resolution):
    """Quantize multiple non-normalized probs to given resolution.

    Args:
      probs: An [N, M]-shaped numpy array of non-normalized probabilities.

    Returns:
      An [N, M]-shaped array of quantized probabilities such that
      np.all(result.sum(axis=1) == resolution).
    """
    assert len(probs.shape) == 2
    N, M = probs.shape
    probs = probs + TINY
    probs /= probs.sum(axis=1, keepdims=True)
    result = np.zeros(probs.shape, np.int8)
    range_N = np.arange(N, dtype=np.int32)
    for _ in range(resolution):
        sample = probs.argmax(axis=1)
        result[range_N, sample] += 1
        probs[range_N, sample] -= 1.0 / resolution
    return result


def make_ragged_index(columns):
    """Make an index to hold data in a ragged array.

    Args:
      columns: A list of [N, _]-shaped numpy arrays of varying size, where
        N is the number of rows.

    Returns:
      A [len(columns) + 1]-shaped array of begin,end positions of each column.
    """
    ragged_index = np.zeros([len(columns) + 1], dtype=np.int32)
    ragged_index[0] = 0
    for v, column in enumerate(columns):
        ragged_index[v + 1] = ragged_index[v] + column.shape[-1]
    return ragged_index


def make_ragged_mask(ragged_index, mask):
    """Convert a boolean mask from dense to ragged format.

    Args:
      ragged_index: A [V+1]-shaped numpy array as returned by
        make_ragged_index.
      mask: A [V,...]-shaped numpy array of booleans.

    Returns:
      A [R,...]-shaped numpy array, where R = ragged_index[-1].
    """
    V = ragged_index.shape[0] - 1
    R = ragged_index[-1]
    assert mask.shape[0] == V
    assert mask.dtype == np.bool_
    ragged_mask = np.empty((R, ) + mask.shape[1:], dtype=np.bool_)
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        ragged_mask[beg:end] = mask[v]
    return ragged_mask


def guess_counts(ragged_index, data):
    """Guess the multinomial count of each feature.

    This should guess 1 for categoricals and (max - min) for ordinals.

    Args:
      ragged_index: A [V+1]-shaped numpy array as returned by
        make_ragged_index.
      data: A [N, R]-shaped ragged array of multinomial count data, where
        N is the number of rows and R = ragged_index[-1].

    Returns:
      A [V]-shaped array of multinomial totals.
    """
    V = len(ragged_index) - 1
    counts = np.zeros(V, np.int8)
    for v in range(V):
        beg, end = ragged_index[v:v + 2]
        counts[v] = data[:, beg:end].sum(axis=1).max()
    return counts


POOL = None


def parallel_map(fun, args):
    global POOL
    args = list(args)
    if len(args) < 2:
        return list(map(fun, args))
    if POOL is None:
        POOL = multiprocessing.Pool()
    return POOL.map(fun, args)


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
