# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/contrib/funsor"):
            try:
                import funsor
                funsor.set_backend("torch")
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="funsor is not installed"))
            if "stage" not in item.keywords:
                item.add_marker(pytest.mark.stage("funsor"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
