# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

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
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", category=RuntimeWarning)
                actual = model()
            for w in ws:
                if w.category == RuntimeWarning and "falling back to default" in str(w):
                    pytest.skip("overwriting initial value")
                else:
                    warnings.warn(str(w.message), category=w.category)

            assert_close(actual, init_value)
