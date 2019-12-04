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


@pytest.mark.parametrize("df", [3.9, 9.1])
@pytest.mark.parametrize("dim", [1, 2])
def test_rsample(dim, df, num_samples=200 * 1000):
    scale_tril = (0.5 * torch.randn(dim)).exp().diag() + 0.1 * torch.randn(dim, dim)
    scale_tril = scale_tril.tril(0)
    scale_tril.requires_grad_(True)

    d = MultivariateStudentT(torch.tensor(df), torch.zeros(dim), scale_tril)
    z = d.rsample(sample_shape=(num_samples,))
    loss = z.pow(2.0).sum(-1).mean()
    loss.backward()

    actual_scale_tril_grad = scale_tril.grad.data.clone()
    scale_tril.grad.zero_()

    analytic = (df / (df - 2.0)) * torch.mm(scale_tril, scale_tril.t()).diag().sum()
    analytic.backward()
    expected_scale_tril_grad = scale_tril.grad.data

    assert_equal(expected_scale_tril_grad, actual_scale_tril_grad, prec=0.1)
