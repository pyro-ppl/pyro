import pytest
import torch

from pyro.ops.ssm_gp import MaternKernel
from tests.common import assert_equal


@pytest.mark.parametrize('num_gps', [1, 2, 3])
@pytest.mark.parametrize('nu', [1.5, 2.5])
def test_matern_kernel(num_gps, nu):
    mk = MaternKernel(nu=nu, num_gps=num_gps, log_length_scale_init=0.1 * torch.randn(num_gps))

    dt = torch.rand(1).item()
    forward = mk.transition_matrix(dt)
    backward = mk.transition_matrix(-dt)
    forward_backward = torch.matmul(forward, backward)

    # going forward dt in time and then backward dt in time should bring us back to the identity
    assert_equal(forward_backward,
                 torch.eye(mk.state_dim).unsqueeze(0).expand(num_gps, mk.state_dim, mk.state_dim))

    # let's just check that these are PSD
    mk.stationary_covariance().cholesky()
    mk.process_covariance(forward).cholesky()
