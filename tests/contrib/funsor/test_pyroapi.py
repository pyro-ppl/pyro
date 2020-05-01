# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401

# triggers backend registration
import pyro.contrib.funsor  # noqa: F401


@pytest.fixture(params=["contrib.funsor"])
def backend(request):
    with pyro_backend(request.param):
        yield
