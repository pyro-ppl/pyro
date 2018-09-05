from __future__ import absolute_import, division, print_function

import torch
import scipy.special as sc
import pytest
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import Trace_ELBO, TraceEnum_ELBO
from pyro.contrib.oed.eig import (
    vi_ape, naive_rainforth_eig, donsker_varadhan_eig, barber_agakov_ape
)
from pyro.contrib.oed.util import get_indices, lexpand
from tests.common import assert_equal
# TODO move the following code out of examples, e.g. to contrib.lmer
from examples.contrib.oed.models.bayes_linear import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix, analytic_posterior_cov,
    normal_inverse_gamma_linear_model, normal_inverse_gamma_guide, group_linear_model,
    group_normal_guide, sigmoid_model, rf_group_assignments
)
from examples.contrib.oed.guides.amort import (
    LinearModelGuide, GuideDV
)


#########################################################################################
# Designs
#########################################################################################
# All design tensors have shape: batch x n x p
# AB test
AB_test_2d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])

# Design on S^1
item_thetas_small = torch.linspace(0., np.pi/2, 5).unsqueeze(-1)
X_circle_5d_1n_2p = torch.stack([item_thetas_small.cos(), -item_thetas_small.sin()], dim=-1)

# Random effects designs
AB_test_reff_6d_10n_12p, AB_sigmoid_design_6d = rf_group_assignments(10)

#########################################################################################
# Models
#########################################################################################
# Linear models
basic_2p_linear_model_sds_10_2pt5, basic_2p_guide = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]))
_, basic_2p_guide_w1 = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]), coef_label="w1")
basic_2p_ba_guide = lambda d: LinearModelGuide(d, {"w": 2}).guide  # noqa: E731
group_2p_linear_model_sds_10_2pt5 = group_linear_model(torch.tensor(0.), torch.tensor([10.]), torch.tensor(0.),
                                                       torch.tensor([2.5]), torch.tensor(1.))
group_2p_guide = group_normal_guide(torch.tensor(1.), (1,), (1,))
group_2p_ba_guide = lambda d: LinearModelGuide(d, {"w1": 1, "w2": 1}).guide  # noqa: E731

########################################################################################
# Aux
########################################################################################

elbo = TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss


def linear_model_ground_truth(model, design, observation_labels, target_labels, eig=True):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    posterior_covs = [analytic_posterior_cov(prior_cov, x, model.obs_sd) for x in torch.unbind(design)]
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_posterior_covs = [S[target_indices, :][:, target_indices] for S in posterior_covs]
    if eig:
        prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
        return prior_entropy - torch.tensor([0.5*torch.logdet(2*np.pi*np.e*C) for C in target_posterior_covs])
    else:
        return torch.tensor([0.5*torch.logdet(2*np.pi*np.e*C) for C in target_posterior_covs])


def lm_H_prior(model, design, observation_labels, target_labels):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_prior_covs = prior_cov[target_indices, :][:, target_indices]
    return 0.5*torch.logdet(2*np.pi*np.e*target_prior_covs)


def mean_field_guide(batch_tensor, design):
    # A batched variable
    w_p = pyro.param("w_p", 0.2*torch.ones(batch_tensor.shape))
    u_p = pyro.param("u_p", 0.5*torch.ones(batch_tensor.shape))
    pyro.sample("w", dist.Bernoulli(w_p))
    pyro.sample("u", dist.Bernoulli(u_p))


def basic_model(batch_tensor, design):
    pyro.sample("w", dist.Bernoulli(design[0]*torch.ones(batch_tensor.shape)))
    pyro.sample("u", dist.Bernoulli(design[1]*torch.ones(batch_tensor.shape)))
    pyro.sample("y", dist.Delta(torch.tensor([1.])))


def h(p):
    return -(sc.xlogy(p, p) + sc.xlog1py(1 - p, -p))

@pytest.mark.skip
@pytest.mark.parametrize("model,arg,design,guide,expected_ape,n_steps", [
    # Test without running any steps- should get entropy at initialization value
    (basic_model, torch.Tensor([0.0]), torch.Tensor([0.3, 0.4]),
     mean_field_guide, torch.Tensor([h(0.2)+h(0.5)]), 0),
    # These two learn the design (50 steps sufficient for SVI)
    (basic_model, torch.Tensor([0.0]), torch.Tensor([0.3, 0.4]),
     mean_field_guide, torch.Tensor([h(0.3)+h(0.4)]), 50),
    (basic_model, torch.Tensor([0.0]), torch.Tensor([0.3, 0.5]),
     mean_field_guide, torch.Tensor([h(0.3)+h(0.5)]), 50)
])
def test_ape_svi(model, arg, design, guide, expected_ape, n_steps):
    # Reset seed: deals with noise in SVI etc
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    vi_parameters = {
        "guide": lambda d: guide(arg, d),
        "optim": optim.Adam({"lr": 0.01}),
        "loss": Trace_ELBO(),
        "num_steps": n_steps
    }
    is_parameters = {"num_samples": 1}
    ape = vi_ape(lambda d: model(arg, d), design, "y", ["w", "u"], vi_parameters,
                 is_parameters)
    assert_equal(ape, expected_ape, prec=1e-4)

# turn into-
#def test_eig_independent(model, design, observation_labels, target_labels, estimator, args, expected):



@pytest.mark.parametrize("model,design,observation_labels,target_labels,estimator,args,eig", [
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w", 
        naive_rainforth_eig, [2000, 2000], True),
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
        vi_ape, [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}),
        "loss": elbo, "num_steps": 1500}, {"num_samples": 1}], False),
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
        donsker_varadhan_eig, [400, 3200, GuideDV(basic_2p_ba_guide(5)),
        optim.Adam({"lr": 0.025}), False, None, 500], True),
    (basic_2p_linear_model_sds_10_2pt5, AB_test_2d_10n_2p, "y", "w",
        barber_agakov_ape, [20, 400, basic_2p_ba_guide(2), optim.Adam({"lr": 0.05}),
        False, None, 500], False),
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
        barber_agakov_ape, [20, 400, basic_2p_ba_guide(5), optim.Adam({"lr": 0.05}),
        False, None, 500], False)
])
def test_eig_lm(model, design, observation_labels, target_labels, estimator, args, eig):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    y = estimator(model, design, observation_labels, target_labels, *args)
    y_true = linear_model_ground_truth(model, design, observation_labels, target_labels, eig=eig)
    print(y)
    print(y_true)
    print(estimator.__name__)
    percent_error = torch.max(torch.abs(y - y_true) / torch.abs(y_true))
    assert percent_error < 0.1


# Also group test with xfail for vi_ape

