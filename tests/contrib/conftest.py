# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/contrib"):
            if "stage" not in item.keywords:
                item.add_marker(pytest.mark.stage("integration_batch_1"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
