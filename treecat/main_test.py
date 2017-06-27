from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import treecat.__main__ as main


@pytest.mark.parametrize('engine', ['numpy'])
def test_profile_train(engine):
    main.profile_train(10, 10, engine=engine)


@pytest.mark.parametrize('engine', ['numpy'])
def test_profile_serve(engine):
    main.profile_serve(10, 10, engine=engine)
