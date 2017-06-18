from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import treecat.__main__ as main


def test_profile_fit():
    main.profile_fit(10, 10)
