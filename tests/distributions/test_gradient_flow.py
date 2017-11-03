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


@pytest.mark.xfail(reason="TODO: clamp logits to ensure finite values")
@pytest.mark.parametrize('init_tensor_type', [torch.FloatTensor])
def test_bernoulli_with_logits_underflow_gradient(init_tensor_type):
    p = Variable(init_tensor_type([-1e40]), requires_grad=True)
    bernoulli = Bernoulli(logits=p)
    log_pdf = bernoulli.batch_log_pdf(Variable(init_tensor_type([0])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)


@pytest.mark.xfail(reason="TODO: clamp logits to ensure finite values")
@pytest.mark.parametrize('init_tensor_type', [torch.DoubleTensor, torch.FloatTensor])
def test_bernoulli_with_logits_overflow_gradient(init_tensor_type):
    p = Variable(init_tensor_type([1e40]), requires_grad=True)
    bernoulli = Bernoulli(logits=p)
    log_pdf = bernoulli.batch_log_pdf(Variable(init_tensor_type([1])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)


@pytest.mark.parametrize('init_tensor_type', [torch.DoubleTensor, torch.FloatTensor])
def test_categorical_gradient(init_tensor_type):
    p = Variable(init_tensor_type([0, 1]), requires_grad=True)
    categorical = Categorical(p)
    log_pdf = categorical.batch_log_pdf(Variable(init_tensor_type([0, 1])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)


@pytest.mark.parametrize('init_tensor_type', [torch.DoubleTensor, torch.FloatTensor])
def test_categorical_gradient_with_logits(init_tensor_type):
    p = Variable(init_tensor_type([-float('inf'), 0]), requires_grad=True)
    categorical = Categorical(logits=p)
    log_pdf = categorical.batch_log_pdf(Variable(init_tensor_type([0, 1])))
    log_pdf.sum().backward()
    assert_equal(log_pdf.data[0], 0)
    assert_equal(p.grad.data[0], 0)
