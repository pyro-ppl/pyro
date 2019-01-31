from __future__ import absolute_import, division, print_function

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.lkj import corr_cholesky_constraint
from tests.common import assert_tensors_equal


@pytest.mark.parametrize("value_shape", [(1, 1), (3, 1, 1), (3, 3), (1, 3, 3), (5, 3, 3)])
def test_constraint(value_shape):
    value = torch.randn(value_shape).tril()
    value.diagonal(dim1=-2, dim2=-1).exp_()
    value = value / value.norm(2, dim=-1, keepdim=True)

    assert_tensors_equal(corr_cholesky_constraint.check(value), torch.ones(value_shape[:-2]))
