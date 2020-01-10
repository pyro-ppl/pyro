# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.contrib.oed.eig import (
    nmc_eig, posterior_eig, marginal_eig, marginal_likelihood_eig, vnmc_eig, lfire_eig,
    donsker_varadhan_eig)
from pyro.contrib.util import iter_plates_to_shape

from tests.common import assert_equal

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


@pytest.fixture
def finite_space_model():
    def model(design):
        batch_shape = design.shape
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            theta = pyro.sample("theta", dist.Bernoulli(.4).expand(batch_shape))
            y = pyro.sample("y", dist.Bernoulli((design + theta) / 2.))
            return y
    return model


@pytest.fixture
def one_point_design():
    return torch.tensor(.5)


@pytest.fixture
def true_eig():
    return torch.tensor(0.12580366909478014)


def posterior_guide(y_dict, design, observation_labels, target_labels):

    y = torch.cat(list(y_dict.values()), dim=-1)
    a, b = pyro.param("a", torch.tensor(0.)), pyro.param("b", torch.tensor(0.))
    pyro.sample("theta", dist.Bernoulli(logits=a + b*y))


def marginal_guide(design, observation_labels, target_labels):

    logit_p = pyro.param("logit_p", torch.tensor(0.))
    pyro.sample("y", dist.Bernoulli(logits=logit_p))


def likelihood_guide(theta_dict, design, observation_labels, target_labels):

    theta = torch.cat(list(theta_dict.values()), dim=-1)
    a, b = pyro.param("a", torch.tensor(0.)), pyro.param("b", torch.tensor(0.))
    pyro.sample("y", dist.Bernoulli(logits=a + b*theta))


def make_lfire_classifier(n_theta_samples):
    def lfire_classifier(design, trace, observation_labels, target_labels):
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)
        a, b = pyro.param("a", torch.zeros(n_theta_samples)), pyro.param("b", torch.zeros(n_theta_samples))

        return a + b*y

    return lfire_classifier


def dv_critic(design, trace, observation_labels, target_labels):
    y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
    y = torch.cat(list(y_dict.values()), dim=-1)
    theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
    theta = torch.cat(list(theta_dict.values()), dim=-1)

    w_y = pyro.param("w_y", torch.tensor(0.))
    w_theta = pyro.param("w_theta", torch.tensor(0.))
    w_ytheta = pyro.param("w_ytheta", torch.tensor(0.))

    return y*w_y + theta*w_theta + y*theta*w_ytheta


########################################################################################################################
# TESTS
########################################################################################################################


def test_posterior_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    posterior_eig(finite_space_model, one_point_design, "y", "theta", num_samples=10,
                  num_steps=250, guide=posterior_guide,
                  optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = posterior_eig(finite_space_model, one_point_design, "y", "theta", num_samples=10,
                                  num_steps=250, guide=posterior_guide,
                                  optim=optim.Adam({"lr": 0.01}), final_num_samples=1000)
    assert_equal(estimated_eig, true_eig, prec=1e-2)


def test_marginal_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    marginal_eig(finite_space_model, one_point_design, "y", "theta", num_samples=10,
                 num_steps=250, guide=marginal_guide,
                 optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = marginal_eig(finite_space_model, one_point_design, "y", "theta", num_samples=10,
                                 num_steps=250, guide=marginal_guide,
                                 optim=optim.Adam({"lr": 0.01}), final_num_samples=1000)
    assert_equal(estimated_eig, true_eig, prec=1e-2)


def test_marginal_likelihood_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    marginal_likelihood_eig(finite_space_model, one_point_design, "y", "theta", num_samples=10,
                            num_steps=250, marginal_guide=marginal_guide, cond_guide=likelihood_guide,
                            optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = marginal_likelihood_eig(finite_space_model, one_point_design, "y", "theta", num_samples=10,
                                            num_steps=250, marginal_guide=marginal_guide, cond_guide=likelihood_guide,
                                            optim=optim.Adam({"lr": 0.01}), final_num_samples=1000)
    assert_equal(estimated_eig, true_eig, prec=1e-2)


@pytest.mark.xfail(reason="Bernoullis are not reparametrizable and current VNMC implementation "
                          "assumes reparametrization")
def test_vnmc_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    # Pre-train (large learning rate)
    vnmc_eig(finite_space_model, one_point_design, "y", "theta", num_samples=[9, 3],
             num_steps=250, guide=posterior_guide,
             optim=optim.Adam({"lr": 0.1}))
    # Finesse (small learning rate)
    estimated_eig = vnmc_eig(finite_space_model, one_point_design, "y", "theta", num_samples=[9, 3],
                             num_steps=250, guide=posterior_guide,
                             optim=optim.Adam({"lr": 0.01}), final_num_samples=[1000, 100])
    assert_equal(estimated_eig, true_eig, prec=1e-2)


def test_nmc_eig_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = nmc_eig(finite_space_model, one_point_design, "y", "theta", M=40, N=40 * 40)
    assert_equal(estimated_eig, true_eig, prec=1e-2)


def test_lfire_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = lfire_eig(finite_space_model, one_point_design, "y", "theta", num_y_samples=5,
                              num_theta_samples=50, num_steps=1000, classifier=make_lfire_classifier(50),
                              optim=optim.Adam({"lr": 0.0025}), final_num_samples=500)
    assert_equal(estimated_eig, true_eig, prec=1e-2)


def test_dv_finite_space_model(finite_space_model, one_point_design, true_eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    donsker_varadhan_eig(finite_space_model, one_point_design, "y", "theta", num_samples=100,
                         num_steps=250, T=dv_critic, optim=optim.Adam({"lr": 0.1}))
    estimated_eig = donsker_varadhan_eig(finite_space_model, one_point_design, "y", "theta", num_samples=100,
                                         num_steps=250, T=dv_critic, optim=optim.Adam({"lr": 0.01}),
                                         final_num_samples=2000)
    assert_equal(estimated_eig, true_eig, prec=1e-2)
