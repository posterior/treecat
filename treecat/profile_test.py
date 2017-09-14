from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from treecat.profile import eval
from treecat.profile import serve
from treecat.profile import train


@pytest.mark.parametrize('parallel', [True, False])
def test_profile_train(parallel):
    train(10, 10, parallel=parallel)


def test_profile_serve():
    serve(10, 10)


def test_profile_eval():
    eval(10, 10)
