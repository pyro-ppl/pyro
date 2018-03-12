from __future__ import absolute_import, division, print_function

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/perf"):
            if "perf" not in item.keywords:
                item.add_marker(pytest.mark.stage("perf"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
