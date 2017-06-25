from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import treecat.__main__ as main


@pytest.mark.parametrize('engine', ['numpy', 'tensorflow'])
def test_profile_train(engine):
    main.profile_train(10, 10, engine=engine)
