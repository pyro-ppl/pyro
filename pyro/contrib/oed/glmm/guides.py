# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from contextlib import ExitStack

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.util import (
    tensor_to_dict, rmv, rvv, rtril, lexpand, iter_plates_to_shape
)
from pyro.ops.linalg import rinverse


class LinearModelPosteriorGuide(nn.Module):

    def __init__(self, d, w_sizes, y_sizes, regressor_init=0., scale_tril_init=3., use_softplus=True, **kwargs):
        """
        Guide for linear models. No amortisation happens over designs.
        Amortisation over data is taken care of by analytic formulae for
        linear models (heavy use of truth).

        :param tuple d: the shape by which to expand the guide parameters, e.g. `(num_batches, num_designs)`.
        :param dict w_sizes: map from variable string names to int, indicating the dimension of each
                             weight vector in the linear model.
        :param float regressor_init: initial value for the regressor matrix used to learn the posterior mean.
        :param float scale_tril_init: initial value for posterior `scale_tril` parameter.
        :param bool use_softplus: whether to transform the regressor by a softplus transform: useful if the
                                  regressor should be nonnegative but close to zero.
        """
        super().__init__()
        # Represent each parameter group as independent Gaussian
        # Making a weak mean-field assumption
        # To avoid this- combine labels
        if not isinstance(d, (tuple, list, torch.Tensor)):
            d = (d,)
        self.regressor = nn.ParameterDict({l: nn.Parameter(
                regressor_init * torch.ones(*(d + (p, sum(y_sizes.values()))))) for l, p in w_sizes.items()})
        self.scale_tril = nn.ParameterDict({l: nn.Parameter(
                scale_tril_init * lexpand(torch.eye(p), *d)) for l, p in w_sizes.items()})
        self.w_sizes = w_sizes
        self.use_softplus = use_softplus
        self.softplus = nn.Softplus()

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        return self.linear_model_formula(y, design, target_labels)

    def linear_model_formula(self, y, design, target_labels):

        if self.use_softplus:
            mu = {l: rmv(self.softplus(self.regressor[l]), y) for l in target_labels}
        else:
            mu = {l: rmv(self.regressor[l], y) for l in target_labels}
        scale_tril = {l: rtril(self.scale_tril[l]) for l in target_labels}

        return mu, scale_tril

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("posterior_guide", self)

        # Returns two dicts from labels -> tensors
        mu, scale_tril = self.get_params(y_dict, design, target_labels)

        for l in target_labels:
            w_dist = dist.MultivariateNormal(mu[l], scale_tril=scale_tril[l])
            pyro.sample(l, w_dist)


class LinearModelLaplaceGuide(nn.Module):
    """
    Laplace approximation for a (G)LM.

    :param tuple d: the shape by which to expand the guide parameters, e.g. `(num_batches, num_designs)`.
    :param dict w_sizes: map from variable string names to int, indicating the dimension of each
                         weight vector in the linear model.
    :param str tau_label: the label used for inverse variance parameter sample site, or `None` to indicate a
                          fixed variance.
    :param float init_value: initial value for the posterior mean parameters.
    """
    def __init__(self, d, w_sizes, tau_label=None, init_value=0.1, **kwargs):
        super().__init__()
        # start in train mode
        self.train()
        if not isinstance(d, (tuple, list, torch.Tensor)):
            d = (d,)
        self.means = nn.ParameterDict()
        if tau_label is not None:
            w_sizes[tau_label] = 1
        for l, mu_l in tensor_to_dict(w_sizes, init_value*torch.ones(*(d + (sum(w_sizes.values()), )))).items():
            self.means[l] = nn.Parameter(mu_l)
        self.scale_trils = {}
        self.w_sizes = w_sizes

    @staticmethod
    def _hessian_diag(y, x, event_shape):
        batch_shape = x.shape[:-len(event_shape)]
        assert tuple(x.shape) == tuple(batch_shape) + tuple(event_shape)

        dy = torch.autograd.grad(y, [x, ], create_graph=True)[0]
        H = []

        # collapse independent dimensions into a single one,
        # and dependent dimensions into another single one
        batch_size = 1
        for batch_shape_dim in batch_shape:
            batch_size *= batch_shape_dim

        event_size = 1
        for event_shape_dim in event_shape:
            event_size *= event_shape_dim

        flat_dy = dy.reshape(batch_size, event_size)

        # loop over dependent part
        for i in range(flat_dy.shape[-1]):
            dyi = flat_dy.index_select(-1, torch.tensor([i]))
            Hi = torch.autograd.grad([dyi], [x, ], grad_outputs=[torch.ones_like(dyi)], retain_graph=True)[0]
            H.append(Hi)
        H = torch.stack(H, -1).reshape(*(x.shape + event_shape))
        return H

    def finalize(self, loss, target_labels):
        """
        Compute the Hessian of the parameters wrt ``loss``

        :param torch.Tensor loss: the output of evaluating a loss function such as
                                  `pyro.infer.Trace_ELBO().differentiable_loss` on the model, guide and design.
        :param list target_labels: list indicating the sample sites that are targets, i.e. for which information gain
                                   should be measured.
        """
        # set self.training = False
        self.eval()
        for l, mu_l in self.means.items():
            if l not in target_labels:
                continue
            hess_l = self._hessian_diag(loss, mu_l, event_shape=(self.w_sizes[l],))
            cov_l = rinverse(hess_l)
            self.scale_trils[l] = cov_l.cholesky(upper=False)

    def forward(self, design, target_labels=None):
        """
        Sample the posterior.

        :param torch.Tensor design: tensor of possible designs.
        :param list target_labels: list indicating the sample sites that are targets, i.e. for which information gain
                                   should be measured.
        """
        if target_labels is None:
            target_labels = list(self.means.keys())

        pyro.module("laplace_guide", self)
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(design.shape[:-2]):
                stack.enter_context(plate)

            if self.training:
                # MAP via Delta guide
                for l in target_labels:
                    w_dist = dist.Delta(self.means[l]).to_event(1)
                    pyro.sample(l, w_dist)
            else:
                # Laplace approximation via MVN with hessian
                for l in target_labels:
                    w_dist = dist.MultivariateNormal(self.means[l], scale_tril=self.scale_trils[l])
                    pyro.sample(l, w_dist)


class SigmoidGuide(LinearModelPosteriorGuide):

    def __init__(self, d, n, w_sizes, **kwargs):
        super().__init__(d, w_sizes, **kwargs)
        self.inverse_sigmoid_scale = nn.Parameter(torch.ones(n))
        self.h1_weight = nn.Parameter(torch.ones(n))
        self.h1_bias = nn.Parameter(torch.zeros(n))

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)

        # Approx invert transformation on y in expectation
        y, y1m = y.clamp(1e-35, 1), (1.-y).clamp(1e-35, 1)
        logited = y.log() - y1m.log()
        y_trans = logited/.1
        y_trans = y_trans * self.inverse_sigmoid_scale
        hidden = self.softplus(y_trans)
        y_trans = y_trans + hidden * self.h1_weight + self.h1_bias

        return self.linear_model_formula(y_trans, design, target_labels)


class NormalInverseGammaGuide(LinearModelPosteriorGuide):

    def __init__(self, d, w_sizes, mf=False, tau_label="tau", alpha_init=100.,
                 b0_init=100., **kwargs):
        super().__init__(d, w_sizes, **kwargs)
        self.alpha = nn.Parameter(alpha_init*torch.ones(d))
        self.b0 = nn.Parameter(b0_init*torch.ones(d))
        self.mf = mf
        self.tau_label = tau_label

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)

        coefficient_labels = [label for label in target_labels if label != self.tau_label]
        mu, scale_tril = self.linear_model_formula(y, design, coefficient_labels)
        mu_vec = torch.cat(list(mu.values()), dim=-1)

        yty = rvv(y, y)
        ytxmu = rvv(y, rmv(design, mu_vec))
        beta = self.b0 + .5*(yty - ytxmu)

        return mu, scale_tril, self.alpha, beta

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("ba_guide", self)

        mu, scale_tril, alpha, beta = self.get_params(y_dict, design, target_labels)

        if self.tau_label in target_labels:
            tau_dist = dist.Gamma(alpha, beta)
            tau = pyro.sample(self.tau_label, tau_dist)
            obs_sd = 1./tau.sqrt().unsqueeze(-1).unsqueeze(-1)

        for label in target_labels:
            if label != self.tau_label:
                if self.mf:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label])
                else:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label]*obs_sd)
                pyro.sample(label, w_dist)


class GuideDV(nn.Module):
    """A Donsker-Varadhan `T` family based on a guide family via
    the relation `T = log p(theta) - log q(theta | y, d)`
    """
    def __init__(self, guide):
        super().__init__()
        self.guide = guide

    def forward(self, design, trace, observation_labels, target_labels):

        trace.compute_log_prob()
        prior_lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        conditional_guide = pyro.condition(self.guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
                y_dict, design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        posterior_lp = sum(cond_trace.nodes[l]["log_prob"] for l in target_labels)

        return posterior_lp - prior_lp
