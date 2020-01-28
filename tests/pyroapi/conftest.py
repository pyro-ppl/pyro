# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_runtest_call(item):
    try:
        item.runtest()
    except NotImplementedError as e:
        pytest.xfail(str(e))
