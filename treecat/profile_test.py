from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from treecat.profile import serve
from treecat.profile import train


def test_profile_serve():
    serve(10, 10)


def test_profile_train():
    train(10, 10)
