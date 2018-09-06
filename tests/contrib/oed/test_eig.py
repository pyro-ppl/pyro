from __future__ import absolute_import, division, print_function

import torch
import scipy.special as sc
import pytest
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import (
    vi_ape, naive_rainforth_eig, donsker_varadhan_eig, barber_agakov_ape
)
from pyro.contrib.oed.util import get_indices
from pyro.contrib.glmm import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix, analytic_posterior_cov,
    group_linear_model, group_normal_guide
)
from pyro.contrib.glmm.guides import LinearModelGuide, GuideDV


#########################################################################################
# Designs
#########################################################################################
# All design tensors have shape: batch x n x p
# AB test
AB_test_2d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])

# Design on S^1
item_thetas_small = torch.linspace(0., np.pi/2, 5).unsqueeze(-1)
X_circle_5d_1n_2p = torch.stack([item_thetas_small.cos(), -item_thetas_small.sin()], dim=-1)

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


def bernoulli_guide(design):
    w_p = pyro.param("w_p", torch.tensor(0.2))
    u_p = pyro.param("u_p", torch.tensor(0.5))
    pyro.sample("w", dist.Bernoulli(w_p.expand(design.shape[:-1])))
    pyro.sample("u", dist.Bernoulli(u_p.expand(design.shape[:-1])))


def bernoulli_ba_guide(y_dict, design, observation_labels, target_labels):
    return bernoulli_guide(design)


def bernoulli_model(design):
    pyro.sample("w", dist.Bernoulli(design[..., 0]))
    pyro.sample("u", dist.Bernoulli(design[..., 1]))
    pyro.sample("y", dist.Delta(torch.tensor([1.])))


def bernoulli_ground_truth(model, design, observation_labels, target_labels, eig=True):
    if eig:
        return torch.tensor(0.)
    else:
        return torch.tensor(h(design[0]) + h(design[1]))


def h(p):
    return -(sc.xlogy(p, p) + sc.xlog1py(1 - p, -p))


@pytest.mark.parametrize("model,design,observation_labels,target_labels,estimator,args,eig,allow_error", [
    (bernoulli_model, torch.tensor([0.3, 0.4]), "y", ["w", "u"], vi_ape,
        [{"guide": bernoulli_guide, "optim": optim.Adam({"lr": 0.01}),
          "loss": elbo, "num_steps": 100}, {"num_samples": 1}], False, 1e-2),
    (bernoulli_model, torch.tensor([0.3, 0.4]), "y", ["w", "u"], naive_rainforth_eig,
        [100, 100], True, 1e-2),
    (bernoulli_model, torch.tensor([0.3, 0.4]), "y", ["w", "u"], barber_agakov_ape,
        [20, 800, bernoulli_ba_guide, optim.Adam({"lr": 0.01}),
         False, None, 1000], False, 1e-2),
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
        naive_rainforth_eig, [500, 500], True, 0.2),
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
        vi_ape,
        [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}),
          "loss": elbo, "num_steps": 1000}, {"num_samples": 1}], False, 0.3),
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
        donsker_varadhan_eig,
        [200, 800, GuideDV(basic_2p_ba_guide(5)),
         optim.Adam({"lr": 0.025}), False, None, 500], True, 0.3),
    (basic_2p_linear_model_sds_10_2pt5, AB_test_2d_10n_2p, "y", "w",
        barber_agakov_ape,
        [20, 400, basic_2p_ba_guide(2), optim.Adam({"lr": 0.05}),
         False, None, 500], False, 0.2),
    (basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
        barber_agakov_ape,
        [20, 400, basic_2p_ba_guide(5), optim.Adam({"lr": 0.05}),
         False, None, 500], False, 0.2),
    (group_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w1",
        naive_rainforth_eig, [400, 400, 400], True, 0.2),
    # This fails because guide is wrong
    pytest.param(
        group_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w1", vi_ape,
        [{"guide": group_2p_guide, "optim": optim.Adam({"lr": 0.05}),
          "loss": elbo, "num_steps": 1000}, {"num_samples": 1}], False, 0.3,
        marks=pytest.mark.xfail),
    (group_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w1",
        donsker_varadhan_eig,
        [200, 400, GuideDV(group_2p_ba_guide(5)),
         optim.Adam({"lr": 0.025}), False, None, 500], True, 0.3),
    (group_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w1",
        barber_agakov_ape,
        [20, 400, group_2p_ba_guide(5), optim.Adam({"lr": 0.05}),
         False, None, 500], False, 0.2)
])
def test_eig_lm(model, design, observation_labels, target_labels, estimator, args, eig, allow_error):
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    y = estimator(model, design, observation_labels, target_labels, *args)
    if model is bernoulli_model:
        y_true = bernoulli_ground_truth(model, design, observation_labels, target_labels, eig=eig)
    else:
        y_true = linear_model_ground_truth(model, design, observation_labels, target_labels, eig=eig)
    print()
    print(estimator.__name__)
    print(y)
    print(y_true)
    error = torch.max(torch.abs(y - y_true))
    assert error < allow_error
