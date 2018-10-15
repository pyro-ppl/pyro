from __future__ import absolute_import, division, print_function

import torch
import numpy as np

import pyro
from pyro.contrib.util import get_indices, lexpand
from pyro.contrib.glmm import analytic_posterior_cov
from pyro.contrib.oed.eig import barber_agakov_ape, vi_ape


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


def mc_H_prior(model, design, observation_labels, target_labels, num_samples=1000):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_samples)
    trace = pyro.poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    return -lp.sum(0)/num_samples


def vi_eig_lm(model, design, observation_labels, target_labels, *args, **kwargs):
    # **Only** applies to linear models - analytic prior entropy
    ape = vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
    return prior_entropy - ape


def ba_eig_lm(model, design, observation_labels, target_labels, *args, **kwargs):
    # **Only** applies to linear models - analytic prior entropy
    ape = barber_agakov_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
    return prior_entropy - ape


def ba_eig_mc(model, design, observation_labels, target_labels, *args, **kwargs):
    # Compute the prior entropy my Monte Carlo, the uses barber_agakov_ape
    if "num_hprior_samples" in kwargs:
        hprior = mc_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = mc_H_prior(model, design, observation_labels, target_labels)
    return hprior - barber_agakov_ape(model, design, observation_labels, target_labels, *args, **kwargs)
