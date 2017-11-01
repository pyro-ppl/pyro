import pytest
import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid

from pyro.distributions import Bernoulli, Categorical
from tests.common import assert_equal


@pytest.mark.parametrize('init_tensor_type', [torch.DoubleTensor, torch.FloatTensor])
def test_bernoulli_underflow_gradient(init_tensor_type):
    p = Variable(init_tensor_type([0]), requires_grad=True)
    bernoulli = Bernoulli(sigmoid(p) * 0.0)
    log_pdf = bernoulli.batch_log_pdf(Variable(init_tensor_type([0])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)


@pytest.mark.parametrize('init_tensor_type', [torch.DoubleTensor, torch.FloatTensor])
def test_bernoulli_overflow_gradient(init_tensor_type):
    p = Variable(init_tensor_type([1e32]), requires_grad=True)
    bernoulli = Bernoulli(sigmoid(p))
    log_pdf = bernoulli.batch_log_pdf(Variable(init_tensor_type([1])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)


@pytest.mark.parametrize('init_tensor_type', [torch.DoubleTensor, torch.FloatTensor])
def test_categorical_gradient(init_tensor_type):
    p = Variable(init_tensor_type([0, 1]), requires_grad=True)
    bernoulli = Categorical(p)
    log_pdf = bernoulli.batch_log_pdf(Variable(init_tensor_type([0, 1])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)
