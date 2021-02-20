# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401

pytestmark = pytest.mark.stage('unit')


@pytest.fixture(params=["pyro", "minipyro"])
def backend(request):
    with pyro_backend(request.param):
        yield
