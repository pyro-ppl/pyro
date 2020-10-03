import pytest
import torch
import torch.tensor as tt

from pyro.distributions import OrderedLogistic


@pytest.mark.parametrize("n_cutpoints", [1, 5, 100])
@pytest.mark.parametrize("n_predictors", [1, 5, 20, 50])
def test_sample(n_cutpoints, n_predictors):
    predictors = torch.randn(n_predictors)
    cutpoints = torch.sort(torch.randn(n_cutpoints)).values
    dist = OrderedLogistic(predictors, cutpoints)
    sample = dist.sample([100])
    assert sample.shape[-1] == n_predictors
    assert sample.min().item() >= 0
    assert sample.max().item() <= n_cutpoints


def test_assertions():
    good_predictors = torch.randn(5)
    good_cutpoints = torch.sort(torch.randn(10)).values
    for pred, cp in [
        (good_predictors, tt([])),
        (good_predictors, tt([[1, 2], [3, 4]])),
        (good_predictors, tt([1, 2, 3, 4, 0])),
        (tt([]), good_cutpoints),
        (tt([[1, 2], [3, 4]]), good_cutpoints),
    ]:
        with pytest.raises(AssertionError):
            OrderedLogistic(pred, cp)
