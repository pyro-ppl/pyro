# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/contrib/funsor"):
            if "stage" not in item.keywords:
                item.add_marker(pytest.mark.stage("funsor"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
            if "test_pyroapi" in item.nodeid and "test_mean_field_ok" in item.nodeid:
                item.add_marker(pytest.mark.xfail(reason="not implemented"))
