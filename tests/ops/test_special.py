# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.special import log_beta, log_beta_stirling


@pytest.mark.parametrize("tol", [
    1e-8, 1e-6, 1e-4, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.1, 1.,
])
def test_log_beta_stirling(tol):
    x = torch.logspace(-5, 5, 100)
    y = x.unsqueeze(-1)

    expected = log_beta(x, y)
    actual = log_beta_stirling(x, y, tol=tol)

    assert (actual <= expected).all()
    assert (expected < actual + tol).all()
