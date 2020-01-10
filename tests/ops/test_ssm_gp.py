# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.ssm_gp import MaternKernel
from tests.common import assert_equal


@pytest.mark.parametrize('num_gps', [1, 2, 3])
@pytest.mark.parametrize('nu', [0.5, 1.5, 2.5])
def test_matern_kernel(num_gps, nu):
    mk = MaternKernel(nu=nu, num_gps=num_gps, length_scale_init=0.1 + torch.rand(num_gps))

    dt = torch.rand(1).item()
    forward = mk.transition_matrix(dt)
    backward = mk.transition_matrix(-dt)
    forward_backward = torch.matmul(forward, backward)

    # going forward dt in time and then backward dt in time should bring us back to the identity
    eye = torch.eye(mk.state_dim).unsqueeze(0).expand(num_gps, mk.state_dim, mk.state_dim)
    assert_equal(forward_backward, eye)

    # let's just check that these are PSD
    mk.stationary_covariance().cholesky()
    mk.process_covariance(forward).cholesky()

    # evolving forward infinitesimally should yield the identity
    nudge = mk.transition_matrix(torch.tensor([1.0e-9]))
    assert_equal(nudge, eye)
