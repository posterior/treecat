from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from treecat.serving import TreeCatServer
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import train_model


@pytest.mark.xfail
def test_server_init():
    model = train_model(TINY_DATA, TINY_MASK, TINY_CONFIG)
    TreeCatServer(model['tree'], model['suffstats'], model['config'])
