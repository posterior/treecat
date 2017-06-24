from treecat.cTreecat import echo


def test_echo():
    assert echo('foo') == 'foo'
