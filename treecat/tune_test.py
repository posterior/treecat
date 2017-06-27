from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from treecat.tune import profile_serve
from treecat.tune import profile_train


def test_profile_train():
    profile_train(10, 10)


def test_profile_serve():
    profile_serve(10, 10)
