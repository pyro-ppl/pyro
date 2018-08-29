import torch
from torch import nn
import math

import pyro
import pyro.distributions as dist
from pyro import poutine

from pyro.contrib.oed.util import rmv, rvv, rinverse, rdiag, rtril


class Ba_lm_guide(nn.Module):

    def __init__(self, p, d, w_sizes):
        super(Ba_lm_guide, self).__init__()
        self.regu = nn.Parameter(-2.*torch.ones(p))
        self.scale_tril = nn.Parameter(3.*torch.ones(d, p, p))
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()

    def forward(self, y, design, target_label):

        # TODO fix this
        design = design[..., :self.w_sizes[target_label]]

        anneal = torch.diag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = rinverse(xtx)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y))

        scale_tril = rtril(self.scale_tril)

        return mu, scale_tril

    def guide(self, y_dict, design, observation_labels, target_labels):

        target_label = target_labels[0]
        pyro.module("ba_guide", self)

        y = y_dict["y"]
        mu, scale_tril = self.forward(y, design, target_label)

        # guide distributions for w
        w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        pyro.sample(target_label, w_dist)


class Ba_sigmoid_guide(nn.Module):

    def __init__(self, p, d, n, w_sizes):
        super(Ba_sigmoid_guide, self).__init__()
        self.inverse_sigmoid_scale = nn.Parameter(torch.ones(n))
        self.h1_weight = nn.Parameter(torch.ones(n))
        self.h1_bias = nn.Parameter(torch.zeros(n))
        self.scale_tril = nn.Parameter(10.*torch.ones(d, p, p))
        self.regu = nn.Parameter(-2.*torch.ones(d, p))
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()

    def forward(self, y, design, target_label):

        # Approx invert transformation on y in expectation
        y, y1m = y.clamp(1e-35, 1), (1.-y).clamp(1e-35, 1)
        logited = y.log() - y1m.log()
        y_trans = logited/.1
        y_trans = y_trans * self.inverse_sigmoid_scale
        hidden = self.softplus(y_trans)
        y_trans = y_trans + hidden * self.h1_weight + self.h1_bias

        # TODO fix this
        design = design[..., :self.w_sizes[target_label]]

        anneal = rdiag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = rinverse(xtx)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y_trans))

        scale_tril = rtril(self.scale_tril)

        return mu, scale_tril

    def guide(self, y_dict, design, observation_labels, target_labels):

        target_label = target_labels[0]
        pyro.module("ba_guide", self)

        y = y_dict["y"]
        mu, scale_tril = self.forward(y, design, target_label)

        # guide distributions for w
        w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        pyro.sample(target_label, w_dist)

        return mu, scale_tril


class Ba_nig_guide(nn.Module):

    def __init__(self, p, d, w_sizes, mf=False):
        super(Ba_nig_guide, self).__init__()
        self.regu = nn.Parameter(-2.*torch.ones(p))
        self.scale_tril = nn.Parameter(10.*torch.ones(d, p, p))
        self.alpha = nn.Parameter(100.*torch.ones(d))
        self.b0 = nn.Parameter(100.*torch.ones(d))
        self.w_sizes = w_sizes
        self.mf = mf
        self.softplus = nn.Softplus()

    def forward(self, y, design, target_label):

        # TODO fix this
        design = design[..., :self.w_sizes[target_label]]

        anneal = torch.diag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = rinverse(xtx)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y))

        scale_tril = rtril(self.scale_tril)

        yty = rvv(y, y)
        xtymu = torch.matmul(y.unsqueeze(-2), design).matmul(mu.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        beta = self.b0 + .5*(yty - xtymu)

        return mu, scale_tril, self.alpha, beta

    def guide(self, y_dict, design, observation_labels, target_labels):

        target_label = target_labels[0]
        pyro.module("ba_guide", self)

        y = y_dict["y"]
        mu, scale_tril, alpha, beta = self.forward(y, design, target_label)

        tau_dist = dist.Gamma(alpha, beta)
        tau = pyro.sample("tau", tau_dist)
        obs_sd = 1./tau.sqrt().unsqueeze(-1).unsqueeze(-1)

        # guide distributions for w
        if self.mf:
            w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        else:
            w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril*obs_sd)
        pyro.sample(target_label, w_dist)


class GuideDV(nn.Module):
    def __init__(self, guide):
        super(GuideDV, self).__init__()
        self.guide = guide

    def forward(self, design, trace, observation_labels, target_labels):
        # TODO fix this
        observation_label = observation_labels[0]
        target_label = target_labels[0]

        trace.compute_log_prob()
        prior_lp = trace.nodes[target_label]["log_prob"] 
        y_dict = {observation_label: trace.nodes[observation_label]["value"]}
        theta_dict = {target_label: trace.nodes[target_label]["value"]}

        conditional_guide = pyro.condition(self.guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
                y_dict, design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        posterior_lp = cond_trace.nodes[target_label]["log_prob"]

        return posterior_lp - prior_lp
