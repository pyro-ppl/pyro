import pytest


def pytest_runtest_call(item):
    try:
        item.runtest()
    except NotImplementedError as e:
        pytest.xfail(str(e))
