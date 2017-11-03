from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

from pyro.nn.clipped_nn import ClippedSigmoid, ClippedSoftmax
from tests.common import assert_equal


@pytest.mark.parametrize('Tensor', [torch.FloatTensor, torch.DoubleTensor])
def test_clipped_softmax(Tensor):
    epsilon = 1e-5
    try:
        clipped_softmax = ClippedSoftmax(epsilon, dim=1)
    except TypeError:
        # Support older pytorch 0.2 release.
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
    try:
        clipped_sigmoid = ClippedSigmoid(epsilon, dim=1)
    except TypeError:
        # Support older pytorch 0.2 release.
        clipped_sigmoid = ClippedSigmoid(epsilon)
    ps = Variable(Tensor([0, 1]))
    sigmoid_ps = clipped_sigmoid(ps)
    print("epsilon = {}".format(epsilon))
    print("sigmoid_ps = {}".format(sigmoid_ps))
    assert (sigmoid_ps.data >= epsilon).all()
    assert (sigmoid_ps.data <= 1 - epsilon).all()
