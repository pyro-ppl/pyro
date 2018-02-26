import torch

from torch.autograd import Variable
import pytest

from pyro.infer.util import MultiViewTensor
from tests.common import assert_equal


def test_multiview_tensor_add():
    x = MultiViewTensor()
    x.add(Variable(torch.ones(2, 3, 4)))
    assert_equal(x[torch.Size([2, 3, 4])].data, torch.ones(2, 3, 4))
    x.add(x)
    x.add(Variable(torch.ones(2, 3, 5)))
    assert_equal(x[torch.Size([2, 3, 4])].data, 2.0 * torch.ones(2, 3, 4))
    x.add(x)
    assert_equal(x[torch.Size([2, 3, 4])].data, 4.0 * torch.ones(2, 3, 4))
    assert_equal(x[torch.Size([2, 3, 5])].data, 2.0 * torch.ones(2, 3, 5))
    y = MultiViewTensor(Variable(1.5 * torch.ones(2, 3, 4)))
    x.add(y)
    assert_equal(x[torch.Size([2, 3, 4])].data, 5.5 * torch.ones(2, 3, 4))


def test_multiview_tensor_sum_leftmost_all_but():
    x = MultiViewTensor(Variable(1.5 * torch.ones(2, 3, 4)))
    x.add(Variable(2.5 * torch.ones(1, 3, 4)))
    x.add(Variable(3.5 * torch.ones(2, 3, 1)))
    result = x.sum_leftmost_all_but(2)
    assert_equal(result[torch.Size([3, 4])].data, 5.5 * torch.ones(3, 4))
    assert_equal(result[torch.Size([3, 1])].data, 7.0 * torch.ones(3, 1))


def test_multiview_tensor_contract_to():
    x = MultiViewTensor(Variable(1.5 * torch.ones(2, 1, 4)))
    x.add(Variable(2.5 * torch.ones(1, 3, 4)))
    x.add(Variable(3.5 * torch.ones(2, 3, 1)))
    x.add(Variable(4.5 * torch.ones(3, 1)))
    result214 = x.contract_to(torch.ones(2, 1, 4)).data
    result134 = x.contract_to(torch.ones(1, 3, 4)).data
    result231 = x.contract_to(torch.ones(2, 3, 1)).data
    assert_equal(result214, (1.5 + 7.5 + 10.5 + 13.5) * torch.ones(2, 1, 4))
    assert_equal(result134, (2.5 + 3.0 + 7.0 + 4.5) * torch.ones(1, 3, 4))
    assert_equal(result231, (3.5 + 10.0 + 6.0 + 4.5) * torch.ones(2, 3, 1))
    with pytest.raises(ValueError):
        x.contract_to(torch.ones(3, 1))
