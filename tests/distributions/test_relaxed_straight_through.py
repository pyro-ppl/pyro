from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.autograd import grad

import pyro.distributions as dist
from tests.common import assert_equal

PROBS = [
	[0.25, 0.75],
	[0.25, 0.5, 0.25],
	[[0.25, 0.75], [0.75, 0.25]],
	[[[0.25, 0.75]], [[0.75, 0.25]]],
	[0.1] * 10,
]


@pytest.mark.parametrize('probs', PROBS)
def test_shapes(probs):
	temperature = torch.tensor(0.5)
	probs = torch.tensor(probs, requires_grad=True)
	d = dist.RelaxedCategoricalStraightThrough(temperature, probs=probs)
	sample = d.rsample()
	log_prob = d.log_prob(sample)
	grad_probs = grad(log_prob.sum(), [probs])[0]
	#import pdb as pdb; pdb.set_trace()
	assert grad_probs.shape == probs.shape


@pytest.mark.xfail(reason='numerical approximation to categorical when reducing temperature does not hold up')
@pytest.mark.parametrize('probs', PROBS)
def test_temperature(probs):
	temperature = torch.tensor(0.02)
	probs = torch.tensor(probs, requires_grad=True)
	d = dist.RelaxedCategoricalStraightThrough(temperature, probs=probs)
	d2 = dist.OneHotCategorical(probs=probs)
	sample = d.rsample()
	assert (sample.sum(-1) == 1).all(), 'not one-hot: {}'.format(sample)
	log_prob = d.log_prob(sample)
	log_prob2 = d2.log_prob(sample)
	assert_equal(log_prob, log_prob2, prec=0.01, msg='{} vs {}'.format(log_prob, log_prob2))