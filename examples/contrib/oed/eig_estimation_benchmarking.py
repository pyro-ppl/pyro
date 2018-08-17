import time
import torch
import pytest
import numpy as np
from functools import partial

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import (
    vi_ape, naive_rainforth, donsker_varadhan_loss
)

from models.bayes_linear import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix, analytic_posterior_entropy,
    bayesian_linear_model, normal_inv_gamma_family_guide, normal_inverse_gamma_linear_model,
    normal_inverse_gamma_guide
)
from dv.neural import T_neural

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
- Barber-Agakov (with amortization)
"""

# design tensors have shape: batch x n x p
X_lm = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in range(0, 11)])
X_small = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])
item_thetas = torch.linspace(0., 2*np.pi, 10).unsqueeze(-1)
X_circle = torch.stack([item_thetas.cos(), -item_thetas.sin()], dim=-1)


def vi_for_group_lm(design, w1_sd, w2_sd, num_vi_steps, num_is_samples):
    model = partial(bayesian_linear_model, 
                    w_means={"w1": torch.tensor(0.), "w2": torch.tensor(0.)},
                    w_sqrtlambdas={"w1": 1./w1_sd, "w2": 1./w2_sd},
                    obs_sd=torch.tensor(1.))
    guide = partial(normal_inv_gamma_family_guide,
                    w_sizes={"w1": w1_sd.shape, "w2": w2_sd.shape},
                    obs_sd=torch.tensor(1.))
    prior_cov = torch.diag(w1_sd**2)
    H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
    return H_prior - vi_ape(
        model,
        design,
        observation_labels="y",
        target_labels="w1",
        vi_parameters={
            "guide": guide,
            "optim": optim.Adam({"lr": 0.05}),
            "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
            "num_steps": num_vi_steps},
        is_parameters={"num_samples": num_is_samples})


def vi_for_lm(design, w_sds, num_vi_steps, num_is_samples, lr=0.05, known_cov=True):
    if known_cov:
        model, guide = zero_mean_unit_obs_sd_lm(w_sds)
        prior_cov = torch.diag(w_sds**2)
        H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
    else:
        alpha = torch.tensor(5.)
        beta = torch.tensor(4.)
        model = normal_inverse_gamma_linear_model(torch.tensor(0.), w_sds, 
                                                  alpha, beta)
        guide = normal_inverse_gamma_guide(w_sds.shape)
        prior_cov = torch.diag(w_sds**2)
        H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov) + torch.digamma(alpha) \
                  - 2*torch.log(beta) + alpha + torch.lgamma(alpha) \
                  + (1. - alpha)*torch.digamma(alpha)
    
    return H_prior - vi_ape(
        model,
        design,
        observation_labels="y",
        vi_parameters={
            "guide": guide,
            "optim": optim.Adam({"lr": lr}),
            "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
            "num_steps": num_vi_steps},
        is_parameters={"num_samples": num_is_samples})


def lm_true_ape(X_lm, w_sds, obs_sd=torch.tensor(1.)):
    prior_cov = torch.diag(w_sds**2)
    designs = torch.unbind(X_lm)
    true_ape = [analytic_posterior_entropy(prior_cov, x, obs_sd) for x in designs]
    return torch.tensor(true_ape)


def lm_true_eig(X_lm, w_sds, obs_sd=torch.tensor(1.)):
    prior_cov = torch.diag(w_sds**2)
    H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
    return H_prior - lm_true_ape(X_lm, w_sds, obs_sd)


def naive_rainforth_lm(X, w_sds, N, M, known_cov=True):
    if known_cov:
        model, _ = zero_mean_unit_obs_sd_lm(w_sds)
    else:
        model = normal_inverse_gamma_linear_model(torch.tensor(0.), w_sds, 
                                                  torch.tensor(5.), torch.tensor(4.))
    return naive_rainforth(model, X, "y", None, N=N, M=M)


def naive_rainforth_group_lm(X, w1_sd, w2_sd, N, M):
    model = partial(bayesian_linear_model, 
                    w_means={"w1": torch.tensor(0.), "w2": torch.tensor(0.)},
                    w_sqrtlambdas={"w1": 1./w1_sd, "w2": 1./w2_sd},
                    obs_sd=torch.tensor(1.))
    return naive_rainforth(model, X, "y", "w1", N=N, M=M, M_prime=M)


def donsker_varadhan_lm(X, w_sds, n_iter, n_samples, lr, T,
                        final_X=None, final_n_samples=None, return_history=False):
    model, _ = zero_mean_unit_obs_sd_lm(w_sds)
    if final_X is None:
        final_X = X
    if final_n_samples is None:
        final_n_samples = n_samples
    dv_loss_fn = donsker_varadhan_loss(model, "y", T)
    params = None
    opt = optim.ClippedAdam({"lr": lr, "betas": (0.92, 0.999)})
    history = []
    for step in range(n_iter):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        agg_loss, dv_loss = dv_loss_fn(X, n_samples)
        agg_loss.backward()
        if return_history:
            history.append(dv_loss)
        params = [pyro.param(name).unconstrained()
                  for name in pyro.get_param_store().get_all_param_names()]
        opt(params)
    _, dv_loss = dv_loss_fn(final_X, final_n_samples)
    if return_history:
        return torch.stack(history), dv_loss
    else:
        return dv_loss


@pytest.mark.parametrize("title,arglist", [
    ("A/B testing with unknown covariance",
      # Warning! Guide is not mean-field
     [(X_lm, vi_for_lm, [torch.tensor([10, 2.5]), 5000, 10, 0.05, False]),
      (X_lm, naive_rainforth_lm, [torch.tensor([10., 2.5]), 2000, 2000, False])
      ]),
    ("Linear model targeting one parameter",
     [(X_circle[..., :1], lm_true_eig, [torch.tensor([10.])]),
      (X_circle, vi_for_group_lm, [torch.tensor([10.]), torch.tensor([2.5]), 5000, 1]),
      (X_circle, naive_rainforth_group_lm, [torch.tensor([10.]), torch.tensor([2.5]), 200, 200])
      ]),
    ("Linear model with designs on S^1",
     [(X_circle, lm_true_eig, [torch.tensor([10., 2.5])]),
      (X_circle, vi_for_lm, [torch.tensor([10., 2.5]), 5000, 1, 0.01]),
      (X_circle, naive_rainforth_lm, [torch.tensor([10., 2.5]), 2000, 2000]),
      # (X_circle, donsker_varadhan_lm, [torch.tensor([10., 2.5]), 4000, 200, 0.005, T_neural(2, 2)])
      ]),
    ("A/B test linear model known covariance",
     [(X_lm, lm_true_eig, [torch.tensor([10., 2.5])]),
      (X_lm, vi_for_lm, [torch.tensor([10., 2.5]), 5000, 1]),
      (X_lm, naive_rainforth_lm, [torch.tensor([10., 2.5]), 2000, 2000]),
      # (X_lm, donsker_varadhan_lm, [torch.tensor([10., 2.5]), 4000, 200, 0.005, T_neural(2, 2)])
      ]),
    ("A/B test linear model known covariance (different sds)",
     [(X_lm, lm_true_eig, [torch.tensor([10., .1])]),
      (X_lm, vi_for_lm, [torch.tensor([10., .1]), 10000, 1]),
      (X_lm, naive_rainforth_lm, [torch.tensor([10., .1]), 2000, 2000]),
      # (X_lm, donsker_varadhan_lm, [torch.tensor([10., .1]), 4000, 200, 0.005, T_neural(2, 2)])
      ])
])
def test_eig_and_plot(title, arglist):
    """
    Runs a group of EIG estimation tests and plots the estimates on a single set
    of axes. Typically, each test within one `arglist` should estimate the same quantity.
    This is repeated for each `arglist`.
    """
    # pyro.set_rng_seed(42)
    ys = []
    names = []
    for design_tensor, estimator, args in arglist:
        ys.append(time_eig(design_tensor, estimator, *args))
        names.append(estimator.__name__)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        for y in ys:
            plt.plot(y.detach().numpy(), linestyle='None', marker='o', markersize=10)
        plt.title(title)
        plt.legend(names)
        plt.show()


def time_eig(design_tensor, estimator, *args):
    pyro.clear_param_store()

    t = time.time()
    y = estimator(design_tensor, *args)
    elapsed = time.time() - t

    print(estimator.__name__)
    print('estimate', y)
    print('elapsed', elapsed)
    return y


@pytest.mark.parametrize("design,w_sds,dv_params", [
     (X_small, torch.tensor([10., 2.5]), {"n_iter": 2000, "n_samples": 2000, "lr": 0.0005, "T": T_neural(2, 2),
                                          "final_n_samples": 10000}),
])
def test_dv_lm_convergence(design, w_sds, dv_params):
    """
    Produces a convergence plot for a Donsker-Varadhan EIG estimation.
    """
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    truth = lm_true_eig(design, w_sds)
    dv, final = donsker_varadhan_lm(design, w_sds, return_history=True, **dv_params)
    x = np.arange(0, dv.shape[0])
    print("Final est", final, "Truth", truth)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(x, torch.nn.ReLU()(dv.detach()).numpy())

        for true, col in zip(torch.unbind(truth, 0), plt.rcParams['axes.prop_cycle'].by_key()['color']):
            plt.axhline(true.numpy(), color=col)
        plt.show()
