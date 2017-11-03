from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

from pyro.nn.clipped_nn import ClippedSigmoid, ClippedSoftmax
from tests.common import assert_equal


@pytest.mark.parametrize('Tensor', [torch.FloatTensor, torch.DoubleTensor])
def test_clipped_softmax(Tensor):
    epsilon = 1e-5
    clipped_softmax = ClippedSoftmax(epsilon)
    ps = Variable(Tensor([[0, 1]]))
    softmax_ps = clipped_softmax(ps)
    print("epsilon = {}".format(epsilon))
    print("softmax_ps = {}".format(softmax_ps))
    assert (softmax_ps.data >= epsilon).all()
    assert (softmax_ps.data <= 1 - epsilon).all()
    assert_equal(softmax_ps.data.sum(), 1.0)


@pytest.mark.parametrize('Tensor', [torch.FloatTensor, torch.DoubleTensor])
def test_clipped_sigmoid(Tensor):
    epsilon = 1e-5
    clipped_softmax = ClippedSigmoid(epsilon)
    ps = Variable(Tensor([0, 1]))
    softmax_ps = clipped_softmax(ps)
    print("epsilon = {}".format(epsilon))
    print("softmax_ps = {}".format(softmax_ps))
    assert (softmax_ps.data >= epsilon).all()
    assert (softmax_ps.data <= 1 - epsilon).all()
