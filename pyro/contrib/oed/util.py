from __future__ import absolute_import, division, print_function

import math
import torch

import pyro
from pyro.contrib.util import get_indices, lexpand
from pyro.contrib.glmm import analytic_posterior_cov
from pyro.contrib.oed.eig import posterior_ape, vi_ape, laplace_vi_ape


def linear_model_ground_truth(model, design, observation_labels, target_labels, eig=True):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    design_shape = design.shape
    posterior_covs = [analytic_posterior_cov(prior_cov, x, model.obs_sd) for x in
                      torch.unbind(design.reshape(-1, design_shape[-2], design_shape[-1]))]
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_posterior_covs = [S[target_indices, :][:, target_indices] for S in posterior_covs]
    output = torch.tensor([0.5 * torch.logdet(2 * math.pi * math.e * C)
                           for C in target_posterior_covs])
    if eig:
        prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
        output = prior_entropy - output

    return output.reshape(design.shape[:-2])


def lm_H_prior(model, design, observation_labels, target_labels):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_prior_covs = prior_cov[target_indices, :][:, target_indices]
    return 0.5*torch.logdet(2 * math.pi * math.e * target_prior_covs)


def mc_H_prior(model, design, observation_labels, target_labels, num_samples=1000):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_samples)
    trace = pyro.poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    return -lp.sum(0)/num_samples


def vi_eig_lm(model, design, observation_labels, target_labels, *args, **kwargs):
    """Estimates the EIG by using `vi_ape` to estimate the APE and then computing the prior entropy analytically
    assuming a Gaussian prior. The `model` should have a `w_sds` attribute giving the prior standard deviations of
    the latent variables.

    :param function model: Model (pyro stochastic function) accepting `design` as its only argument.
    :param torch.Tensor design: Tensor of possible designs.
    :param list observation_labels: labels of sample sites regarded as experimental observations.
    :param list target_labels: labels of sample sites regarded as latent variables of interest.
    :param args: passed to `vi_ape`
    :param kwargs: passed to `vi_ape`
    :return: EIG estimate
    :rtype: torch.Tensor
    """
    ape = vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
    return prior_entropy - ape


def vi_eig_mc(model, design, observation_labels, target_labels, *args, **kwargs):
    """Estimates the EIG by using `vi_ape` to estimate the APE and then computing the prior entropy using Monte Carlo.

    :param function model: Model (pyro stochastic function) accepting `design` as its only argument.
    :param torch.Tensor design: Tensor of possible designs.
    :param list observation_labels: labels of sample sites regarded as experimental observations.
    :param list target_labels: labels of sample sites regarded as latent variables of interest.
    :param args: passed to `vi_ape`
    :param kwargs: passed to `vi_ape`
    :return: EIG estimate
    :rtype: torch.Tensor
    """
    if "num_hprior_samples" in kwargs:
        hprior = mc_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = mc_H_prior(model, design, observation_labels, target_labels)
    ape = vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    return hprior - ape


def laplace_vi_eig_mc(model, design, observation_labels, target_labels, *args, **kwargs):
    """Estimates the EIG by using `laplace_vi_ape` to estimate the APE and then computing the prior entropy using
    Monte Carlo.

    :param function model: Model (pyro stochastic function) accepting `design` as its only argument.
    :param torch.Tensor design: Tensor of possible designs.
    :param list observation_labels: labels of sample sites regarded as experimental observations.
    :param list target_labels: labels of sample sites regarded as latent variables of interest.
    :param args: passed to `laplace_vi_ape`
    :param kwargs: passed to `laplace_vi_ape`
    :return: EIG estimate
    :rtype: torch.Tensor
    """
    if "num_hprior_samples" in kwargs:
        hprior = mc_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = mc_H_prior(model, design, observation_labels, target_labels)
    ape = laplace_vi_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    return hprior - ape


def posterior_eig_lm(model, design, observation_labels, target_labels, *args, **kwargs):
    """Estimates the EIG by using `posterior_ape` to estimate the APE and then computing the prior entropy analytically
    assuming a Gaussian prior. The `model` should have a `w_sds` attribute giving the prior standard deviations of
    the latent variables.

    :param function model: Model (pyro stochastic function) accepting `design` as its only argument.
    :param torch.Tensor design: Tensor of possible designs.
    :param list observation_labels: labels of sample sites regarded as experimental observations.
    :param list target_labels: labels of sample sites regarded as latent variables of interest.
    :param args: passed to `posterior_ape`
    :param kwargs: passed to `posterior_ape`
    :return: EIG estimate
    :rtype: torch.Tensor
    """
    ape = posterior_ape(model, design, observation_labels, target_labels, *args, **kwargs)
    prior_entropy = lm_H_prior(model, design, observation_labels, target_labels)
    if isinstance(ape, tuple):
        return tuple(prior_entropy - a for a in ape)
    else:
        return prior_entropy - ape


def posterior_eig_mc(model, design, observation_labels, target_labels, *args, **kwargs):
    """Estimates the EIG by using `posterior_ape` to estimate the APE and then computing the prior entropy using
    Monte Carlo.

    :param function model: Model (pyro stochastic function) accepting `design` as its only argument.
    :param torch.Tensor design: Tensor of possible designs.
    :param list observation_labels: labels of sample sites regarded as experimental observations.
    :param list target_labels: labels of sample sites regarded as latent variables of interest.
    :param args: passed to `posterior_ape`
    :param kwargs: passed to `posterior_ape`
    :return: EIG estimate
    :rtype: torch.Tensor
    """
    if "num_hprior_samples" in kwargs:
        hprior = mc_H_prior(model, design, observation_labels, target_labels, kwargs["num_hprior_samples"])
    else:
        hprior = mc_H_prior(model, design, observation_labels, target_labels)
    return hprior - posterior_ape(model, design, observation_labels, target_labels, *args, **kwargs)
