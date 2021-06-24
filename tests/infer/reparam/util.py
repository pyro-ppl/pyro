# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro import poutine
from pyro.infer.autoguide.initialization import InitMessenger, init_to_value
from pyro.infer.reparam.reparam import Reparam
from tests.common import assert_close


def check_init_reparam(model, reparam):
    assert isinstance(reparam, Reparam)
    with poutine.block():
        init_value = model()
    with InitMessenger(init_to_value(values={"x": init_value})):
        # Sanity check without reparametrizing.
        actual = model()
        assert_close(actual, init_value)

        # Check with reparametrizing.
        with poutine.reparam(config={"x": reparam}):
            actual = model()
            assert_close(actual, init_value)
