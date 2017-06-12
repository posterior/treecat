import contextlib

import pytest


@contextlib.contextmanager
def xfail_if_not_implemented():
    try:
        yield
    except NotImplementedError as e:
        pytest.xfail(reason=str(e))
