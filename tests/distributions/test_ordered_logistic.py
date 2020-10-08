import pytest
import torch
import torch.tensor as tt

from pyro.distributions import OrderedLogistic


@pytest.mark.parametrize("n_cutpoints", [1, 5, 100])
@pytest.mark.parametrize("pred_shape", [(1,), (5,), (5, 5), (1, 2, 3)])
def test_sample(n_cutpoints, pred_shape):
    predictor = torch.randn(pred_shape)
    cutpoints = torch.sort(torch.randn(n_cutpoints)).values
    dist = OrderedLogistic(predictor, cutpoints, validate_args=True)
    sample = dist.sample([100])
    assert sample.shape[1:] == pred_shape
    assert sample.min().item() >= 0
    assert sample.max().item() <= n_cutpoints


def test_constraints():
    predictor = torch.randn(5)
    for cp in (
        tt([1, 2, 3, 4, 0]),
        tt([1, 2, 4, 3, 5]),
        tt([1, 2, 3, 4, 4]),
    ):
        with pytest.raises(ValueError):
            OrderedLogistic(predictor, cp)


def test_broadcast():
    predictor = torch.randn(2, 3, 4)
    # test scenario where `cutpoints.ndim <= predictor.ndim + 1`
    for cp in (
        torch.arange(5),
        torch.arange(5).view(1, -1),
        torch.stack(4*[torch.arange(5)]),
        torch.sort(torch.randn(3, 4, 5), dim=-1).values,
        torch.sort(torch.randn(predictor.shape + (100,)), dim=-1).values,
    ):
        dist = OrderedLogistic(predictor, cp, validate_args=True)
        assert dist.batch_shape == predictor.shape
        assert dist.sample().shape == predictor.shape

    # test scenario where `cutpoints.ndim > predictor.ndim + 1`
    # interpretation is broadcasting batches of cutpoints
    cp = torch.sort(torch.randn(10, 2, 3, 4, 5), dim=-1).values
    dist = OrderedLogistic(predictor, cp, validate_args=True)
    assert dist.batch_shape == (10,) + predictor.shape
    assert dist.sample().shape == (10,) + predictor.shape


def test_expand():
    predictor = torch.randn(4, 5)
    cutpoints = torch.sort(torch.randn(5, 6)).values
    dist = OrderedLogistic(predictor, cutpoints, validate_args=True)
    new_batch_shape = (2, 3, 4, 5)
    dist = dist.expand(new_batch_shape)
    assert dist.batch_shape == torch.Size(new_batch_shape)
    assert dist.event_shape == torch.Size(())
    sample = dist.sample([100])
    assert torch.all(sample <= 6).item()


def test_autograd():
    predictor = torch.randn(5, requires_grad=True)
    cutpoints = torch.sort(torch.randn(3)).values
    cutpoints.requires_grad = True
    data = torch.tensor([0, 1, 2, 3, 0], dtype=float)

    dist = OrderedLogistic(predictor, cutpoints, validate_args=True)
    dist.log_prob(data).sum().backward()

    assert predictor.grad is not None
    assert torch.all(predictor.grad != 0).item()
    assert cutpoints.grad is not None
    assert torch.all(cutpoints.grad != 0).item()