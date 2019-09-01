import pytest

import torch
from torch.distributions import StudentT

from pyro.distributions import MultivariateStudentT
from tests.common import assert_equal


@pytest.mark.parametrize("batch_shape", [
    (),
    (3, 2),
    (1,),
], ids=str)
# FIXME: what is a proper test for log_prob with dim > 1?
@pytest.mark.parametrize("dim", [1])
def test_log_prob(batch_shape, dim):
    loc = torch.randn(batch_shape + (dim,))
    A = torch.randn(batch_shape + (dim, dim + dim))
    scale_tril = A.matmul(A.transpose(-2, -1)).cholesky()
    x = torch.randn(batch_shape + (dim,))
    df = torch.randn(batch_shape).exp() + 2

    actual_log_prob = MultivariateStudentT(df, loc, scale_tril).log_prob(x)
    expected_log_prob = StudentT(df.unsqueeze(-1), loc, scale_tril[..., 0]).log_prob(x).sum(-1)
    assert_equal(actual_log_prob, expected_log_prob)
