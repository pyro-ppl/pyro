import time
from functools import partial
import torch
import pytest
import numpy as np

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_ape, naive_rainforth

from models.bayes_linear import (
    bayesian_linear_model, normal_inv_gamma_guide, group_assignment_matrix,
    analytic_posterior_entropy
)

PLOT = True

########################################################################################
# Linear model with known observation sd
########################################################################################

X_lm = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in range(0, 11)])


def vi_for_lm(design, w_sqrt_lambda, obs_sd, alpha_0, beta_0, num_vi_steps, num_is_samples):
    prior_cov = torch.diag(1./w_sqrt_lambda**2)
    H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
    return H_prior - vi_ape(
        partial(bayesian_linear_model,
                w_mean=torch.tensor(0.),
                w_sqrtlambda=w_sqrt_lambda,
                obs_sd=obs_sd,
                alpha_0=alpha_0,
                beta_0=beta_0),
        design,
        observation_labels="y",
        vi_parameters={
            "guide": partial(normal_inv_gamma_guide,
                             obs_sd=obs_sd),
            "optim": optim.Adam({"lr": 0.05}),
            "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
            "num_steps": num_vi_steps},
        is_parameters={"num_samples": num_is_samples})


def lm_true_ape(X_lm, sqrtlambda, obs_sd):
    prior_cov = torch.diag(1./sqrtlambda**2)
    designs = torch.unbind(X_lm)
    true_ape = [analytic_posterior_entropy(prior_cov, x, obs_sd) for x in designs]
    return torch.tensor(true_ape)


def lm_true_eig(X_lm, sqrtlambda, obs_sd):
    prior_cov = torch.diag(1./sqrtlambda**2)
    H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
    return H_prior - lm_true_ape(X_lm, sqrtlambda, obs_sd)


def naive_rainforth_lm(X_lm, sqrtlambda, obs_sd, alpha_0, beta_0, N, M):
    return naive_rainforth(
        partial(bayesian_linear_model,
                w_mean=torch.tensor(0.),
                w_sqrtlambda=sqrtlambda,
                obs_sd=obs_sd,
                alpha_0=alpha_0,
                beta_0=beta_0),
        X_lm, "y", "w", N=N, M=M)


@pytest.mark.parametrize("arglist", [
     # Warning: do not do this, not a mean-field guide!
     # [(X_lm, vi_for_lm, torch.tensor([.1, .4]), None, torch.tensor(10.), torch.tensor(10.), 5000, 10)],
     [(X_lm, lm_true_eig, [torch.tensor([.1, .4]), torch.tensor(1.)]),
      (X_lm, vi_for_lm, [torch.tensor([.1, .4]), torch.tensor(1.), None, None, 5000, 1]),
      (X_lm, naive_rainforth_lm, [torch.tensor([.1, .4]), torch.tensor(1.), None, None, 2000, 2000])],
     [(X_lm, lm_true_eig, [torch.tensor([.1, 10.]), torch.tensor(1.)]),
      (X_lm, vi_for_lm, [torch.tensor([.1, 10.]), torch.tensor(1.), None, None, 10000, 1]),
      (X_lm, naive_rainforth_lm, [torch.tensor([.1, 10.]), torch.tensor(1.), None, None, 2000, 2000])],
])
def test_eig_and_plot(arglist):
    pyro.set_rng_seed(42)
    ys = []
    for design_tensor, estimator, args in arglist:
        ys.append(time_eig(design_tensor, estimator, *args))

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        for y in ys:
            plt.plot(y.detach().numpy(), linestyle='None', marker='o', markersize=10)
        plt.show()


def time_eig(design_tensor, estimator, *args):
    pyro.clear_param_store()

    t = time.time()
    y = estimator(design_tensor, *args)
    elapsed = time.time() - t

    print(y)
    print(elapsed)
    return y
