from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import logging
import os
import sys
from collections import defaultdict
from timeit import default_timer

PROFILE_TIME = int(os.environ.get('TREECAT_PROFILE_TIME', 0))

LOG_LEVEL = int(os.environ.get('TREECAT_LOG_LEVEL', logging.CRITICAL))
LOG_FILENAME = os.environ.get('TREECAT_LOG_FILE')
LOG_FORMAT = '%(levelname).1s %(name)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, filename=LOG_FILENAME)


def TODO(message=''):
    raise NotImplementedError('TODO {}'.format(message))


class ProfileTimer(object):
    __slots__ = ['elapsed']

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.elapsed -= default_timer()

    def __exit__(self, type, value, traceback):
        self.elapsed += default_timer()


PROFILE_TIMERS = defaultdict(ProfileTimer)


def profile_timed(fun):
    if not PROFILE_TIME:
        return fun
    timer = PROFILE_TIMERS[fun]

    @functools.wraps(fun)
    def profiled_fun(*args, **kwargs):
        with timer:
            return fun(*args, **kwargs)

    return profiled_fun


def print_profile_timers():
    times = [(t.elapsed, f) for (f, t) in PROFILE_TIMERS.items()]
    times.sort(reverse=True)
    sys.stderr.write('{: >10} {}\n'.format('Time (sec)', 'Function'))
    sys.stderr.write('-' * 32 + '\n')
    for time, fun in times:
        if time > 0:
            sys.stderr.write('{: >10.3f} {}.{}\n'.format(
                time, fun.__module__, fun.__name__))


if PROFILE_TIME:
    atexit.register(print_profile_timers)
