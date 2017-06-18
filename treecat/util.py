from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import logging
import os
from collections import defaultdict
from timeit import default_timer

import numpy as np

LOG_LEVEL = int(os.environ.get('TREECAT_LOG_LEVEL', logging.CRITICAL))
PROFILING = (LOG_LEVEL <= 15)
LOG_FILENAME = os.environ.get('TREECAT_LOG_FILE')
LOG_FORMAT = '%(levelname).1s %(name)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, filename=LOG_FILENAME)
logger = logging.getLogger(__name__)


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))


def np_seterr(**settings):
    '''Decorator to run with temporary np.seterr() settings.'''

    def decorator(fun):
        @functools.wraps(fun)
        def decorated_fun(*args, **kwargs):
            old = np.seterr(**settings)
            try:
                return fun(*args, **kwargs)
            finally:
                np.seterr(**old)

        return decorated_fun

    return decorator


class ProfileHistogram(object):
    __slots__ = ['counts']

    def __init__(self):
        self.counts = defaultdict(lambda: 0)

    def add(self, value, delta=1):
        self.counts[value] += delta

    def items(self):
        return self.counts.items()


class ProfileCounter(object):
    __slots__ = ['count']

    def __init__(self):
        self.count = 0

    def add(self, delta=1):
        self.count += delta


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
    timer = PROFILE_TIMERS[fun]

    @functools.wraps(fun)
    def profiled_fun(*args, **kwargs):
        with timer:
            return fun(*args, **kwargs)

    return profiled_fun


# Use these to access global profiling state.
PROFILE_HISTOGRAMS = defaultdict(ProfileHistogram)
PROFILE_COUNTERS = defaultdict(ProfileCounter)
PROFILE_TIMERS = defaultdict(ProfileTimer)


def log_profile_counters():
    histograms = sorted(PROFILE_HISTOGRAMS.items())
    logger.info('-' * 64)
    logger.info('Profile counters:')
    for name, histogram in histograms:
        logger.info('{: >10s} {}'.format('Count', name))
        for value, count in sorted(histogram.items()):
            logger.info('{: >10d} {}'.format(count, value))
    counts = sorted(PROFILE_COUNTERS.items())
    logger.info('{: >10s} {}'.format('Count', 'Counter'))
    for name, counter in counts:
        logger.info('{: >10d} {}'.format(counter.count, name))


def log_profile_timers():
    times = [(t.elapsed, t.count, f) for (f, t) in PROFILE_TIMERS.items()]
    times.sort(reverse=True, key=lambda x: x[0])
    logger.info('-' * 64)
    logger.info('Profile timers:')
    logger.info('{: >10} {: >10} {}'.format('Seconds', 'Calls', 'Function'))
    for time, count, fun in times:
        logger.info('{: >10.3f} {: >10} {}.{}'.format(
            time, count, fun.__module__, fun.__name__))


if PROFILING:
    atexit.register(log_profile_timers)
    atexit.register(log_profile_counters)
