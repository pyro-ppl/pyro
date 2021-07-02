# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    # triggers backend registration
    import funsor

    import pyro.contrib.funsor  # noqa: F401

    funsor.set_backend("torch")
except ImportError:
    pytestmark = pytest.mark.skip()

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401


@pytest.fixture(params=["contrib.funsor"])
def backend(request):
    with pyro_backend(request.param):
        yield
