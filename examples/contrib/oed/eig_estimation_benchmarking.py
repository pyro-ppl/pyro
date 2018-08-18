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

from models.bayes_linear import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix, analytic_posterior_entropy,
    bayesian_linear_model, normal_inv_gamma_family_guide, normal_inverse_gamma_linear_model,
    normal_inverse_gamma_guide, group_linear_model, group_normal_guide
)
from dv.neural import T_neural, T_specialized
from ba.guide import Ba_lm_guide

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
- logistic regression
- LMER with normal response and known obs_sd:
  - aim to learn all unknowns: w, u and G_u
  - aim to learn w and G_u
  - aim to learn w
  - aim to learn u
- logistic-LMER

Estimation techniques:

- analytic EIG, for linear models with known variances
- iterated variational inference
- naive Rainforth (nested Monte Carlo)
- Donsker-Varadhan

TODO:

- VI with amortization
- Barber-Agakov with amortization
- Better guides/ test cases for DV and BA
"""

#########################################################################################
# Designs
#########################################################################################
# All design tensors have shape: batch x n x p
# AB test
AB_test_10d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in range(0, 11)])
AB_test_2d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])

# Design on S^1
item_thetas = torch.linspace(0., np.pi, 10).unsqueeze(-1)
X_circle_10d_1n_2p = torch.stack([item_thetas.cos(), -item_thetas.sin()], dim=-1)

#########################################################################################
# Models
#########################################################################################
# Linear models
basic_2p_linear_model_sds_10_2pt5, basic_2p_guide = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]))
basic_2p_linear_model_sds_10_0pt1, _ = zero_mean_unit_obs_sd_lm(torch.tensor([10., .1]))
group_2p_linear_model_sds_10_2pt5 = group_linear_model(torch.tensor(0.), torch.tensor([10.]), torch.tensor(0.), 
                                                       torch.tensor([2.5]), torch.tensor(1.))
group_2p_guide = group_normal_guide(torch.tensor(1.), (1,), (1,))
nig_2p_linear_model_5_4 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([10., 2.5]),
                                                            torch.tensor([5.]), torch.tensor([4.]))
nig_2p_guide = normal_inverse_gamma_guide((2,))

########################################################################################
# Aux
########################################################################################

elbo = TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss


def linear_model_ground_truth(model, design, observation_labels, target_labels, eig=True):
    out = torch.tensor(0.)
    start = 0
    for label, w_sd in model.w_sds.items():
        if target_labels is None or label in target_labels:
            prior_cov = torch.diag(w_sd**2)
            designs = torch.unbind(design[..., start:(start+w_sd.shape[-1])])
            true_ape = [analytic_posterior_entropy(prior_cov, x, model.obs_sd) for x in designs]
            if eig:
                H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
                out = out + H_prior - torch.tensor(true_ape)
            else:
                # else, APE
                out = out + torch.tensor(true_ape)
        start += w_sd.shape[-1]
    return out


def H_prior(model, design, observation_labels, target_labels):
    out = torch.tensor(0.)
    start = 0
    for label, w_sd in model.w_sds.items():
        if target_labels is None or label in target_labels:
            prior_cov = torch.diag(w_sd**2)
            H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
            out = out + H_prior
        start += w_sd.shape[-1]
    return out


def vi_eig(model, design, observation_labels, target_labels, *args, **kwargs):
    ape = vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = H_prior(model, design, observation_labels, target_labels)
    return prior_entropy - ape


@pytest.mark.parametrize("title,model,design,observation_label,target_label,arglist", [
    ("Linear model targeting one parameter - EIG", 
     group_2p_linear_model_sds_10_2pt5, X_circle_10d_1n_2p, "y", "w1",
     [(linear_model_ground_truth, []),
      (naive_rainforth_eig, [200, 200, 200]),
      (vi_eig,
       [{"guide": group_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 5000}, {"num_samples": 1}])
      ]),
    ("Linear model with designs on S^1",
     basic_2p_linear_model_sds_10_2pt5, X_circle_10d_1n_2p, "y", None,
     [(linear_model_ground_truth, []),
      (vi_eig,
       [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 5000}, {"num_samples": 1}]),
      (naive_rainforth_eig, [2000, 2000]),
      # (X_circle, donsker_varadhan_lm, [torch.tensor([10., 2.5]), 4000, 200, 0.005, T_neural(2, 2)])
      ]),
    ("A/B test linear model known covariance",
     basic_2p_linear_model_sds_10_2pt5, AB_test_10d_10n_2p, "y", None,
     [(linear_model_ground_truth, []),
      (vi_eig,
       [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 5000}, {"num_samples": 1}]),
      (naive_rainforth_eig, [2000, 2000]),
      # (X_lm, donsker_varadhan_lm, [torch.tensor([10., 2.5]), 4000, 200, 0.005, T_neural(2, 2)])
      ]),
    ("A/B test linear model known covariance (different sds)",
     basic_2p_linear_model_sds_10_0pt1, AB_test_10d_10n_2p, "y", None,
     [(linear_model_ground_truth, []),
      (vi_eig,
       [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 5000}, {"num_samples": 1}]),
      (naive_rainforth_eig, [2000, 2000]),
      # (X_lm, donsker_varadhan_lm, [torch.tensor([10., .1]), 4000, 200, 0.005, T_neural(2, 2)])
      ]),
    ("A/B testing with unknown covariance",
     nig_2p_linear_model_5_4, AB_test_10d_10n_2p, "y", None,
      # Warning! Guide is not mean-field
     [(vi_ape,
       [{"guide": nig_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
         "num_steps": 5000}, {"num_samples": 10}]),
      (naive_rainforth_eig, [2000, 2000])
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
    ("Donsker-Varadhan on small AB test",
     basic_2p_linear_model_sds_10_2pt5, AB_test_2d_10n_2p, "y", "w", 
     donsker_varadhan_eig, linear_model_ground_truth,
     {"num_steps": 400, "num_samples": 400, "optim": optim.Adam({"lr": 0.05}),
      "T": T_specialized(), "final_num_samples": 10000}, {}),
    ("Barber-Agakov on small AB test",
     basic_2p_linear_model_sds_10_2pt5, AB_test_2d_10n_2p, "y", "w",
     barber_agakov_ape, linear_model_ground_truth,
     {"num_steps": 400, "num_samples": 10, "optim": optim.Adam({"lr": 0.05}),
      "guide": Ba_lm_guide(torch.tensor([10., 2.5])).guide,
      "final_num_samples": 1000}, {"eig": False})
])
def test_convergence(title, model, design, observation_label, target_label, est1, est2, kwargs1, kwargs2):
    """
    Produces a convergence plot for a Barber-Agakov or Donsker-Varadhan
    EIG estimation.
    """
    t = time.time()
    # pyro.set_rng_seed(42)
    pyro.clear_param_store()
    truth = est2(model, design, observation_label, target_label, **kwargs2)
    dv, final = est1(model, design, observation_label, target_label, return_history=True, **kwargs1)
    x = np.arange(0, dv.shape[0])
    print(est1.__name__)
    print("Final est", final, "Truth", truth, "Error", (final - truth).abs().sum())
    print("Time", time.time() - t)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(x, torch.nn.ReLU()(dv.detach()).numpy())

        for true, col in zip(torch.unbind(truth, 0), plt.rcParams['axes.prop_cycle'].by_key()['color']):
            plt.axhline(true.numpy(), color=col)
        plt.title(title)
        plt.show()
