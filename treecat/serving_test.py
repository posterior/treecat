from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from treecat.serving import TreeCatServer
from treecat.testutil import TINY_CONFIG
from treecat.testutil import TINY_DATA
from treecat.testutil import TINY_MASK
from treecat.training import train_model


def test_server_init():
    model = train_model(TINY_DATA, TINY_MASK, TINY_CONFIG)
    server = TreeCatServer(model['tree'], model['suffstats'], TINY_CONFIG)
    server._get_session(7)
