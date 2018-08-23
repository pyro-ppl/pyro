import time
import torch
import pytest
import numpy as np
from functools import partial

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import (
    vi_ape, naive_rainforth_eig, donsker_varadhan_eig, barber_agakov_ape
)
from pyro.contrib.oed.util import get_indices

from models.bayes_linear import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix, analytic_posterior_cov,
    bayesian_linear_model, normal_inv_gamma_family_guide, normal_inverse_gamma_linear_model,
    normal_inverse_gamma_guide, group_linear_model, group_normal_guide,
    sigmoid_model, rf_group_assignments
)
from dv.neural import T_neural, T_specialized
from ba.guide import Ba_lm_guide, Ba_nig_guide, Ba_sigmoid_guide

PLOT = True

"""
Expected information gain estimation benchmarking
-------------------------------------------------
Models for benchmarking:

- A/B test: linear model with known variances and a discrete design on {0, ..., 10}
- linear model: classical linear model with designs on unit circle
- linear model with two parameter groups, aiming to learn just one
- A/B test with unknown observation covariance:
  - aim to learn regression coefficients *and* obs_sd
  - aim to learn regression coefficients, information on obs_sd ignored
- sigmoid model
- logistic regression*
- LMER with normal response and known obs_sd:*
  - aim to learn all unknowns: w, u and G_u*
  - aim to learn w*
  - aim to learn u*
- logistic-LMER*

* to do

Estimation techniques:

- analytic EIG, for linear models with known variances
- iterated variational inference
- naive Rainforth (nested Monte Carlo)
- Donsker-Varadhan
- Barber-Agakov

TODO:

- better guides- use autoguide, allow different levels of amortization
"""

#########################################################################################
# Designs
#########################################################################################
# All design tensors have shape: batch x n x p
# AB test
AB_test_11d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in range(0, 11)])
AB_test_2d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])

# Design on S^1
item_thetas = torch.linspace(0., np.pi, 10).unsqueeze(-1)
X_circle_10d_1n_2p = torch.stack([item_thetas.cos(), -item_thetas.sin()], dim=-1)
item_thetas_small = torch.linspace(0., np.pi/2, 5).unsqueeze(-1)
X_circle_5d_1n_2p = torch.stack([item_thetas_small.cos(), -item_thetas_small.sin()], dim=-1)

# Random effects designs
AB_test_reff_5d_10n_12p, AB_sigmoid_design_5d = rf_group_assignments(10)
sub_design = AB_test_reff_5d_10n_12p[..., :, 0:2]

#########################################################################################
# Models
#########################################################################################
# Linear models
basic_2p_linear_model_sds_10_2pt5, basic_2p_guide = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]))
basic_2p_linear_model_sds_10_0pt1, _ = zero_mean_unit_obs_sd_lm(torch.tensor([10., .1]))
basic_2p_ba_guide = lambda d: Ba_lm_guide((2,), (d, 3), {"w": 2}).guide
group_2p_linear_model_sds_10_2pt5 = group_linear_model(torch.tensor(0.), torch.tensor([10.]), torch.tensor(0.), 
                                                       torch.tensor([2.5]), torch.tensor(1.))
group_2p_guide = group_normal_guide(torch.tensor(1.), (1,), (1,))
group_2p_ba_guide = lambda d: Ba_lm_guide((1,), (d, 1), {"w1": 1, "w2": 1}).guide
nig_2p_linear_model_3_2 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, .4]),
                                                            torch.tensor([3.]), torch.tensor([2.]))
nig_2p_linear_model_15_14 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, .4]),
                                                              torch.tensor([15.]), torch.tensor([14.]))

nig_2p_guide = normal_inverse_gamma_guide((2,), mf=True)
nig_2p_ba_guide = lambda d: Ba_nig_guide((2,), (d, 3), (d,), {"w": 2}).guide
nig_2p_ba_mf_guide = lambda d: Ba_nig_guide((2,), (d, 3), (d,), {"w": 2}, mf=True).guide

alpha = torch.ones(10)
beta = 10.*torch.ones(10)
k = torch.ones(10)
sigmoid_12p_model = sigmoid_model(torch.tensor(0.), torch.tensor([10., 2.5]), torch.tensor(0.), torch.tensor([1.]*5 + [10.]*5),
                                  torch.tensor(1.), alpha, beta, AB_sigmoid_design_5d)
sigmoid_ba_guide = lambda d: Ba_sigmoid_guide(torch.tensor([10., 2.5]), 5, 10, {"w1": 2}).guide

########################################################################################
# Aux
########################################################################################

elbo = TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss


def linear_model_ground_truth(model, design, observation_labels, target_labels, eig=True):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    designs = torch.unbind(design)
    posterior_covs = [analytic_posterior_cov(prior_cov, x, model.obs_sd) for x in torch.unbind(design)]
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_posterior_covs = [S[target_indices, :][:, target_indices] for S in posterior_covs]
    if eig:
        prior_entropy = H_prior(model, design, observation_labels, target_labels)
        return prior_entropy - torch.tensor([0.5*torch.logdet(2*np.pi*np.e*C) for C in target_posterior_covs])
    else:
        return torch.tensor([0.5*torch.logdet(2*np.pi*np.e*C) for C in target_posterior_covs])


def H_prior(model, design, observation_labels, target_labels):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_prior_covs = prior_cov[target_indices, :][:, target_indices]
    return 0.5*torch.logdet(2*np.pi*np.e*target_prior_covs)


def vi_eig(model, design, observation_labels, target_labels, *args, **kwargs):
    # **Only** applies to linear models - analytic prior entropy
    ape = vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = H_prior(model, design, observation_labels, target_labels)
    return prior_entropy - ape


def ba_eig(model, design, observation_labels, target_labels, *args, **kwargs):
    # **Only** applies to linear models - analytic prior entropy
    ape = barber_agakov_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = H_prior(model, design, observation_labels, target_labels)
    return prior_entropy - ape


@pytest.mark.parametrize("title,model,design,observation_label,target_label,arglist", [
    # ("A/B test linear model targetting one coefficient",
    #  group_2p_linear_model_sds_10_2pt5, AB_test_2d_10n_2p, "y", "w1",
    #  [(linear_model_ground_truth, [False]),
    #   (barber_agakov_ape, [20, 400, group_2p_ba_guide(2), optim.Adam({"lr": 0.05}),
    #     False, None, 500])
    #   ]),
    ("Sigmoid model: 2 classes of participants (5/5), A/B test (5/5)",
     sigmoid_12p_model, AB_test_reff_5d_10n_12p, "y", "w1",
     [(naive_rainforth_eig, [2000, 2000, 2000]),
      (barber_agakov_ape, [20, 800, sigmoid_ba_guide(5), optim.Adam({"lr": 0.05}),
        False, None, 500])
      ]),
    ("A/B testing with unknown covariance (Gamma(15, 14))",
     nig_2p_linear_model_15_14, AB_test_11d_10n_2p, "y", ["w", "tau"],
     [(naive_rainforth_eig, [2000, 2000]),
      (vi_ape,
       [{"guide": nig_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 1000}, {"num_samples": 4}]),
      (barber_agakov_ape, [20, 800, nig_2p_ba_guide(11), optim.Adam({"lr": 0.05}),
        False, None, 500]),
      (barber_agakov_ape, [20, 800, nig_2p_ba_mf_guide(11), optim.Adam({"lr": 0.05}),
        False, None, 500])
      ]),
    ("A/B testing with unknown covariance (Gamma(3, 2))",
     nig_2p_linear_model_3_2, AB_test_11d_10n_2p, "y", ["w", "tau"],
     [(naive_rainforth_eig, [2000, 2000]),
      (vi_ape,
       [{"guide": nig_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 1000}, {"num_samples": 4}]),
      (barber_agakov_ape, [20, 800, nig_2p_ba_guide(11), optim.Adam({"lr": 0.05}),
        False, None, 500]),
      (barber_agakov_ape, [20, 800, nig_2p_ba_mf_guide(11), optim.Adam({"lr": 0.05}),
        False, None, 500])
      ]),
    ("A/B test linear model known covariance",
     basic_2p_linear_model_sds_10_2pt5, AB_test_11d_10n_2p, "y", "w",
     [(linear_model_ground_truth, []),
      (naive_rainforth_eig, [2000, 2000]),
      (vi_eig,
       [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 1000}, {"num_samples": 1}]),
      (donsker_varadhan_eig, [400, 400, T_specialized((11, 3)), optim.Adam({"lr": 0.05}),
        False, None, 500]),
      (ba_eig, [20, 400, basic_2p_ba_guide(11), optim.Adam({"lr": 0.05}),
        False, None, 500])
      ]),
    ("Linear model targeting one parameter", 
     group_2p_linear_model_sds_10_2pt5, X_circle_10d_1n_2p, "y", "w1",
     [(linear_model_ground_truth, []),
      (naive_rainforth_eig, [200, 200, 200]),
      (vi_eig,
       [{"guide": group_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 1000}, {"num_samples": 1}]),
      (donsker_varadhan_eig, [400, 400, T_specialized((10, 3)), optim.Adam({"lr": 0.05}),
        False, None, 500]),
      (ba_eig, [20, 400, group_2p_ba_guide(10), optim.Adam({"lr": 0.05}),
        False, None, 500])
      ]),
    ("Linear model with designs on S^1",
     basic_2p_linear_model_sds_10_2pt5, X_circle_10d_1n_2p, "y", "w",
     [(linear_model_ground_truth, []),
      (naive_rainforth_eig, [2000, 2000]),
      (vi_eig,
       [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 1000}, {"num_samples": 1}]),
      (donsker_varadhan_eig, [400, 400, T_specialized((10, 3)), optim.Adam({"lr": 0.05}),
        False, None, 500]),
      (ba_eig, [20, 400, basic_2p_ba_guide(10), optim.Adam({"lr": 0.05}),
        False, None, 500])
      ]),
    ("A/B test linear model known covariance (different sds)",
     basic_2p_linear_model_sds_10_0pt1, AB_test_11d_10n_2p, "y", "w",
     [(linear_model_ground_truth, []),
      (naive_rainforth_eig, [2000, 2000]),
      (vi_eig,
       [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 1000}, {"num_samples": 1}]),
      (donsker_varadhan_eig, [400, 400, T_specialized((11, 3)), optim.Adam({"lr": 0.05}),
        False, None, 500]),
      (ba_eig, [20, 400, basic_2p_ba_guide(11), optim.Adam({"lr": 0.05}),
        False, None, 500])
      ])
])
def test_eig_and_plot(title, model, design, observation_label, target_label, arglist):
    """
    Runs a group of EIG estimation tests and plots the estimates on a single set
    of axes. Typically, each test within one `arglist` should estimate the same quantity.
    This is repeated for each `arglist`.
    """
    # pyro.set_rng_seed(42)
    ys = []
    names = []
    for estimator, args in arglist:
        ys.append(time_eig(estimator, model, design, observation_label, target_label, args))
        names.append(estimator.__name__)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        for y in ys:
            plt.plot(y.detach().numpy(), linestyle='None', marker='o', markersize=10)
        plt.title(title)
        plt.legend(names)
        plt.show()


def time_eig(estimator, model, design, observation_label, target_label, args):
    pyro.clear_param_store()

    t = time.time()
    y = estimator(model, design, observation_label, target_label, *args)
    elapsed = time.time() - t

    print(estimator.__name__)
    print('estimate', y)
    print('elapsed', elapsed)
    return y


@pytest.mark.parametrize("title,model,design,observation_label,target_label,est1,est2,kwargs1,kwargs2", [
    ("Barber-Agakov on sigmoid",
     sigmoid_12p_model, AB_test_reff_5d_10n_12p, "y", "w1",
     barber_agakov_ape, None,
     {"num_steps": 500, "num_samples": 1000, "optim": optim.Adam({"lr": 0.05}),
      "guide": sigmoid_ba_guide(5), "final_num_samples": 500}, {}),
    ("Barber-Agakov on A/B test with unknown covariance",
     nig_2p_linear_model_3_2, AB_test_2d_10n_2p, "y", "w",
     barber_agakov_ape, None,
     {"num_steps": 800, "num_samples": 20, "optim": optim.Adam({"lr": 0.05}),
      "guide": nig_2p_ba_guide(2), "final_num_samples": 1000}, {}),
    ("Barber-Agakov on A/B test with unknown covariance (mean-field guide)",
     nig_2p_linear_model_3_2, AB_test_2d_10n_2p, "y", "w",
     barber_agakov_ape, None,
     {"num_steps": 800, "num_samples": 20, "optim": optim.Adam({"lr": 0.05}),
      "guide": nig_2p_ba_mf_guide(2), "final_num_samples": 1000}, {}),
    ("Barber-Agakov on circle",
     basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w",
     barber_agakov_ape, linear_model_ground_truth,
     {"num_steps": 400, "num_samples": 10, "optim": optim.Adam({"lr": 0.05}),
      "guide": basic_2p_ba_guide(5),
      "final_num_samples": 1000}, {"eig": False}),
    ("Barber-Agakov on small AB test",
     basic_2p_linear_model_sds_10_2pt5, AB_test_2d_10n_2p, "y", "w",
     barber_agakov_ape, linear_model_ground_truth,
     {"num_steps": 400, "num_samples": 10, "optim": optim.Adam({"lr": 0.05}),
      "guide": basic_2p_ba_guide(2),
      "final_num_samples": 1000}, {"eig": False}),
    ("Donsker-Varadhan on small AB test",
     basic_2p_linear_model_sds_10_2pt5, AB_test_2d_10n_2p, "y", "w", 
     donsker_varadhan_eig, linear_model_ground_truth,
     {"num_steps": 400, "num_samples": 100, "optim": optim.Adam({"lr": 0.05}),
      "T": T_specialized((2, 3)), "final_num_samples": 10000}, {}),
    ("Donsker-Varadhan on circle",
     basic_2p_linear_model_sds_10_2pt5, X_circle_5d_1n_2p, "y", "w", 
     donsker_varadhan_eig, linear_model_ground_truth,
     {"num_steps": 400, "num_samples": 400, "optim": optim.Adam({"lr": 0.05}),
      "T": T_specialized((5, 3)), "final_num_samples": 10000}, {})
])
def test_convergence(title, model, design, observation_label, target_label, est1, est2, kwargs1, kwargs2):
    """
    Produces a convergence plot for a Barber-Agakov or Donsker-Varadhan
    EIG estimation.
    """
    t = time.time()
    # pyro.set_rng_seed(42)
    pyro.clear_param_store()
    if est2 is not None:
        truth = est2(model, design, observation_label, target_label, **kwargs2)
    else:
        truth = None
    dv, final = est1(model, design, observation_label, target_label, return_history=True, **kwargs1)
    x = np.arange(0, dv.shape[0])
    print(est1.__name__)
    if truth is not None:
        print("Final est", final, "Truth", truth, "Error", (final - truth).abs().sum())
    else:
        print("Final est", final)
    print("Time", time.time() - t)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(x, dv.detach().numpy())

        if truth is not None:
            for true, col in zip(torch.unbind(truth, 0), plt.rcParams['axes.prop_cycle'].by_key()['color']):
                plt.axhline(true.numpy(), color=col)
        
        plt.title(title)
        plt.show()
