# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/perf"):
            if "perf" not in item.keywords:
                item.add_marker(pytest.mark.stage("perf"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
