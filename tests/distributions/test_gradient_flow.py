import torch
from torch.autograd import Variable
from pyro.distributions import Bernoulli, Categorical
from torch.nn.functional import sigmoid

from tests.common import assert_equal


def test_bernoulli_underflow_gradient():
    p = Variable(torch.Tensor([0]), requires_grad=True)
    bernoulli = Bernoulli(sigmoid(p) * 0.0)
    log_pdf = bernoulli.batch_log_pdf(Variable(torch.Tensor([0])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)


def test_bernoulli_overflow_gradient():
    p = Variable(torch.Tensor([1e32]), requires_grad=True)
    bernoulli = Bernoulli(sigmoid(p))
    log_pdf = bernoulli.batch_log_pdf(Variable(torch.Tensor([1])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)


def test_categorical_gradient():
    p = Variable(torch.Tensor([0, 1]), requires_grad=True)
    bernoulli = Categorical(p)
    log_pdf = bernoulli.batch_log_pdf(Variable(torch.Tensor([0, 1])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)
