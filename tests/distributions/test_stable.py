import pytest
import torch
import pyro.distributions as dist


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)])
def test_shape(sample_shape, batch_shape):
    stability = torch.empty(batch_shape).uniform_(0, 2).requires_grad_()
    skew = torch.empty(batch_shape).uniform_(-1, 1).requires_grad_()
    scale = torch.randn(batch_shape).exp().requires_grad_()
    loc = torch.randn(batch_shape).requires_grad_()

    d = dist.Stable(stability, skew, scale, loc)
    assert d.batch_shape == batch_shape

    x = d.rsample(sample_shape)
    assert x.shape == sample_shape + batch_shape

    x.sum().backward()
