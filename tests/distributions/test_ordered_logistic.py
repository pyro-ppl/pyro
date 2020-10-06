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
