# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import Trace_ELBO
from pyro.contrib.oed.glmm import known_covariance_linear_model
from pyro.contrib.oed.util import linear_model_ground_truth
from pyro.contrib.oed.eig import (
    nmc_eig, posterior_eig, marginal_eig, marginal_likelihood_eig, vnmc_eig, laplace_eig, lfire_eig,
    donsker_varadhan_eig)
from pyro.contrib.util import rmv, rvv
from pyro.contrib.oed.glmm.guides import LinearModelLaplaceGuide
from tests.common import assert_equal


@pytest.fixture
def linear_model():
    return known_covariance_linear_model(coef_means=torch.tensor(0.),
                                         coef_sds=torch.tensor([1., 1.5]),
                                         observation_sd=torch.tensor(1.))


@pytest.fixture
def one_point_design():
    X = torch.zeros(3, 2)
    X[0, 0] = X[1, 1] = X[2, 1] = 1.
    return X


def posterior_guide(y_dict, design, observation_labels, target_labels):

    y = torch.cat(list(y_dict.values()), dim=-1)
    A = pyro.param("A", torch.zeros(2, 3))
    scale_tril = pyro.param("scale_tril", torch.tensor([[1., 0.], [0., 1.5]]),
                            constraint=torch.distributions.constraints.lower_cholesky)
    mu = rmv(A, y)
    pyro.sample("w", dist.MultivariateNormal(mu, scale_tril=scale_tril))


def marginal_guide(design, observation_labels, target_labels):

    mu = pyro.param("mu", torch.zeros(3))
    scale_tril = pyro.param("scale_tril", torch.eye(3),
                            constraint=torch.distributions.constraints.lower_cholesky)
    pyro.sample("y", dist.MultivariateNormal(mu, scale_tril))


def likelihood_guide(theta_dict, design, observation_labels, target_labels):

    theta = torch.cat(list(theta_dict.values()), dim=-1)
    centre = rmv(design, theta)

    # Need to avoid name collision here
    mu = pyro.param("mu_l", torch.zeros(3))
    scale_tril = pyro.param("scale_tril_l", torch.eye(3),
                            constraint=torch.distributions.constraints.lower_cholesky)

    pyro.sample("y", dist.MultivariateNormal(centre + mu, scale_tril=scale_tril))


# The guide includes some features of the Laplace approximation that would be tiresome to copy across
laplace_guide = LinearModelLaplaceGuide(tuple(), {"w": 2})


def make_lfire_classifier(n_theta_samples):
    def lfire_classifier(design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)

        quadratic_coef = pyro.param("quadratic_coef", torch.zeros(n_theta_samples, 3, 3))
        linear_coef = pyro.param("linear_coef", torch.zeros(n_theta_samples, 3))
        bias = pyro.param("bias", torch.zeros(n_theta_samples))

        y_quadratic = y.unsqueeze(-1) * y.unsqueeze(-2)
        return (quadratic_coef * y_quadratic).sum(-1).sum(-1) + (linear_coef * y).sum(-1) + bias

    return lfire_classifier


def dv_critic(design, trace, observation_labels, target_labels):
    y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
    theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
    x = torch.cat(list(theta_dict.values()) + list(y_dict.values()), dim=-1)

    B = pyro.param("B", torch.zeros(5, 5))
    return rvv(x, rmv(B, x))


########################################################################################################################
# TESTS
########################################################################################################################


def test_posterior_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    posterior_eig(linear_model, one_point_design, "y", "w", num_samples=10,
                  num_steps=250, guide=posterior_guide,
                  optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = posterior_eig(linear_model, one_point_design, "y", "w", num_samples=10,
                                  num_steps=250, guide=posterior_guide,
                                  optim=optim.Adam({"lr": 0.01}), final_num_samples=500)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)


def test_marginal_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    marginal_eig(linear_model, one_point_design, "y", "w", num_samples=10,
                 num_steps=250, guide=marginal_guide,
                 optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = marginal_eig(linear_model, one_point_design, "y", "w", num_samples=10,
                                 num_steps=250, guide=marginal_guide,
                                 optim=optim.Adam({"lr": 0.01}), final_num_samples=500)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)


def test_marginal_likelihood_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    marginal_likelihood_eig(linear_model, one_point_design, "y", "w", num_samples=10,
                            num_steps=250, marginal_guide=marginal_guide, cond_guide=likelihood_guide,
                            optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = marginal_likelihood_eig(linear_model, one_point_design, "y", "w", num_samples=10,
                                            num_steps=250, marginal_guide=marginal_guide, cond_guide=likelihood_guide,
                                            optim=optim.Adam({"lr": 0.01}), final_num_samples=500)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)


def test_vnmc_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    vnmc_eig(linear_model, one_point_design, "y", "w", num_samples=[9, 3],
             num_steps=250, guide=posterior_guide,
             optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = vnmc_eig(linear_model, one_point_design, "y", "w", num_samples=[9, 3],
                             num_steps=250, guide=posterior_guide,
                             optim=optim.Adam({"lr": 0.01}), final_num_samples=[500, 100])
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)


def test_nmc_eig_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = nmc_eig(linear_model, one_point_design, "y", "w", M=60, N=60 * 60)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)


def test_laplace_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # You can use 1 final sample here because linear models have a posterior entropy that is independent of `y`
    estimated_eig = laplace_eig(linear_model, one_point_design, "y", "w",
                                guide=laplace_guide, num_steps=250, final_num_samples=1,
                                optim=optim.Adam({"lr": 0.05}),
                                loss=Trace_ELBO().differentiable_loss)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)


def test_lfire_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = lfire_eig(linear_model, one_point_design, "y", "w", num_y_samples=2, num_theta_samples=50,
                              num_steps=1200, classifier=make_lfire_classifier(50), optim=optim.Adam({"lr": 0.0025}),
                              final_num_samples=100)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)


def test_dv_linear_model(linear_model, one_point_design):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    donsker_varadhan_eig(linear_model, one_point_design, "y", "w", num_samples=100, num_steps=500, T=dv_critic,
                         optim=optim.Adam({"lr": 0.1}))
    estimated_eig = donsker_varadhan_eig(linear_model, one_point_design, "y", "w", num_samples=100,
                                         num_steps=650, T=dv_critic, optim=optim.Adam({"lr": 0.001}),
                                         final_num_samples=2000)
    expected_eig = linear_model_ground_truth(linear_model, one_point_design, "y", "w")
    assert_equal(estimated_eig, expected_eig, prec=5e-2)
