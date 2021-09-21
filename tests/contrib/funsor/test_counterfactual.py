# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pyroapi
import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints

from pyro.ops.indexing import Vindex
from pyro.util import torch_isnan
from tests.common import assert_equal, xfail_param

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    from pyro.contrib.funsor.handlers.counterfactual_messenger import \
        ExogenizeMessenger, FactualWorldMessenger, TwinWorldMessenger

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import handlers, infer, pyro
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)

_PYRO_BACKEND = os.environ.get("TEST_ENUM_PYRO_BACKEND", "contrib.funsor")


@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_normal_counterfactual_smoke():

    # estimand: p(y | do(x)) = \int p(y | z, x) p(x' | z) p(z) dz dx'

    def model():
        #   z
        #  /  \
        # x --> y
        z = pyro.sample("z", dist.Normal(0, 1))
        x = pyro.sample("x", dist.Normal(z, 1))
        x = handlers.primitives.do(x, torch.tensor(1.))
        y = pyro.sample("y", dist.Normal(0.8 * x + 0.3 * z, 1))
        return y

    with ExogenizeMessenger(), TwinWorldMessenger():
        tr = handlers.trace(handlers.condition(cf_model, data={"y": torch.tensor(1.)})).get_trace()

    assert "__CF" in tr.nodes["y"]["funsor"]["value"].inputs


@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_normal_plate_counterfactual_smoke():

    # estimand: p(y | do(x)) = \int p(y | z, x) p(x' | z) p(z) dz dx'

    def model():
        #   z
        #  /  \
        # ---------
        # x --> y |
        # ---------
        z = pyro.sample("z", dist.Normal(0, 1))
        with pyro.plate("data", 3):
            x = pyro.sample("x", dist.Normal(z, 1))
            x = handlers.primitives.do(x, torch.tensor(1.))
            y = pyro.sample("y", dist.Normal(0.8 * x + 0.3 * z, 1))
            return y

    with ExogenizeMessenger(), TwinWorldMessenger():
        tr = handlers.trace(handlers.condition(cf_model, data={"y": torch.tensor([1., 1., 1.])})).get_trace()

    assert "__CF" in tr.nodes["y"]["funsor"]["value"].inputs
