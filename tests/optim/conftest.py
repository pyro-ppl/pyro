# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/optim"):
            if "stage" not in item.keywords:
                item.add_marker(pytest.mark.stage("unit"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))


def pytest_addoption(parser):
    parser.addoption("--plot", action="store", default="FALSE")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.plot != "FALSE"
    if "plot" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("plot", [option_value])
