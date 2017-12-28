from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

from pyro.distributions.empirical import Empirical
from tests.common import assert_equal


def test_empirical_with_strings():
    """
    Creates a set if strings for the distribution and samples from them.
    Then tests if they are scored correctly
    """
    values = ["abc", "def", "gef", "abd"]
    dist = Empirical(values)
    x = dist.sample()
    p = dist.log_pdf(x)
    assert_equal(p, Variable(torch.log(torch.Tensor([.25]))))


@pytest.mark.xfail(reason="NotImplemented")
def test_empirical_deduplicate():
    values = ["abc", "abc", "def", "gef"]
    dist = Empirical(values)
    assert_equal(dist.log_pdf("abc"), Variable(torch.log(torch.Tensor([.5]))))
