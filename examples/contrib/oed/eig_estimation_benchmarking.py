import time
import torch
import pytest
import numpy as np

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import (
    vi_ape, naive_rainforth, donsker_varadhan_loss
)

from models.bayes_linear import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix, analytic_posterior_entropy
)
from nn.donsker_varadhan import DVNeuralNet

PLOT = True

########################################################################################
# Linear model with known observation sd
########################################################################################

X_lm = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in range(0, 11)])
X_small = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])


def vi_for_lm(design, w_sds, num_vi_steps, num_is_samples):
    model, guide = zero_mean_unit_obs_sd_lm(w_sds)
    prior_cov = torch.diag(w_sds**2)
    H_prior = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
    return H_prior - vi_ape(
        model,
        design,
        observation_labels="y",
        vi_parameters={
            "guide": guide,
            "optim": optim.Adam({"lr": 0.05}),
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


def naive_rainforth_lm(X_lm, w_sds, N, M):
    model, _ = zero_mean_unit_obs_sd_lm(w_sds)
    return naive_rainforth(model, X_lm, "y", "w", N=N, M=M)


def donsker_varadhan_lm(X_lm, w_sds, n_iter, n_samples, lr, nn,
                        final_X_lm=None, final_n_samples=None, return_history=False):
    model, _ = zero_mean_unit_obs_sd_lm(w_sds)
    if final_X_lm is None:
        final_X_lm = X_lm
    if final_n_samples is None:
        final_n_samples = n_samples
    dv_loss_fn = donsker_varadhan_loss(model, "y", "w", nn)
    params = None
    opt = optim.Adam({"lr": lr})
    history = []
    for step in range(n_iter):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        agg_loss, dv_loss = dv_loss_fn(X_lm, n_samples)
        agg_loss.backward()
        if return_history:
            history.append(dv_loss)
        params = [pyro.param(name).unconstrained()
                  for name in pyro.get_param_store().get_all_param_names()]
        opt(params)
    _, dv_loss = dv_loss_fn(final_X_lm, final_n_samples)
    if return_history:
        return torch.stack(history), dv_loss
    else:
        return dv_loss


@pytest.mark.parametrize("arglist", [
     [(X_lm, lm_true_eig, [torch.tensor([10., 2.5])]),
      (X_lm, vi_for_lm, [torch.tensor([10., 2.5]), 5000, 1]),
      (X_lm, naive_rainforth_lm, [torch.tensor([10., 2.5]), 2000, 2000]),
      (X_lm, donsker_varadhan_lm, [torch.tensor([10., 2.5]), 4000, 200, 0.005, DVNeuralNet(2, 2)])],
     [(X_lm, lm_true_eig, [torch.tensor([10., .1])]),
      (X_lm, vi_for_lm, [torch.tensor([10., .1]), 10000, 1]),
      (X_lm, naive_rainforth_lm, [torch.tensor([10., .1]), 2000, 2000]),
      (X_lm, donsker_varadhan_lm, [torch.tensor([10., .1]), 4000, 200, 0.005, DVNeuralNet(2, 2)])],
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


@pytest.mark.parametrize("design,w_sds,dv_params", [
     (X_small, torch.tensor([10., 2.5]), {"n_iter": 4000, "n_samples": 200, "lr": 0.005, "nn": DVNeuralNet(2, 2),
                                          "final_n_samples": 10000}),
     (X_small, torch.tensor([10., .1]), {"n_iter": 2000, "n_samples": 200, "lr": 0.005, "nn": DVNeuralNet(2, 2),
                                         "final_n_samples": 10000})
])
def test_dv_lm_convergence(design, w_sds, dv_params):
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
