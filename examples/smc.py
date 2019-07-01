from __future__ import absolute_import, division, print_function

import argparse
import logging

import math
import numpy as onp

import torch
import torch.distributions as td

import matplotlib
import matplotlib.pyplot as plt

# A simple harmonic oscillator with each discretized time step t = 1 
# and mass equal to the spring constant.
#
#   y(t) = (A*z_t + B*eps_t + eps_y)[0]
# 
# where B = [0,0,1], eps_y = [1, 0, 0], eps_t ~ N(0, sigma_eps^2)
# and A is chosen based off Newton's eqns. The prior over sigma is IG(3,1).
def simple_harmonic_model(initial, T):
	A = torch.tensor([[1., 1., -0.5],
				  	  [0., 1., 1.],
				      [-1., 0., 0.]])

	B = torch.tensor([0., 0., 1.])

	prec_eps = torch.distributions.gamma.Gamma(torch.tensor([3.0]),
											   torch.tensor([1.0]))
	prec_y = torch.distributions.gamma.Gamma(torch.tensor([3.0]),
											 torch.tensor([1.0]))
	var_eps = 1.0/prec_eps
	var_y = 1.0/prec_y

	z_t = initial
	y = torch.zeros(T)
	for t in range(T):
		z, y[t] = step(z_t, A, B, var_eps, var_y)


def step(z_t, A, B, var_eps, var_y):
	# Univariate Gaussians
	eps_t = torch.distributions.normal.Normal(torch.tensor([0]), 
											 torch.sqrt(var_eps))
	eps_y = torch.distributions.normal.Normal(torch.tensor([0]), 
											 torch.sqrt(var_y))

	z = A*z_t + B*eps_t
	y = z_t[0] + eps_y

	return z, y

# Compute the target p(z_{1:t}|y_{1:t}) via SMC
# An update step in the SMC procedure includes
# For particle n
# 	1. Draw z^n_t ~ p(z_t|z_{1:t-1}) [t- distributed with parameters alpha, beta, z_{t-1}]
# 	2. Weight w^n_t <- w^n_{t-1} * p(y_t|z^n_t) []
#   3. If effective number of particles(1/ \sum_i (w_i^2)) is below N_thresh
#      then resample proportional to w and set weights to 1/N.
# 	4. Update the hyperparameters of p(theta|z^n_{1:t}) [Add sufficient statisttics to get posterior alpha beta ]
#
# Downdate:
#	1. Downdate the hyperparameter [Keep track of suffiicient statistics and remove alpha, beta]
#   2. Divide by p(y_t|z_t) [Keep track of likelihood values at each time step.] 
#
# Online Version: We could use complicated update/downdate schedules
def inference():
	pass
