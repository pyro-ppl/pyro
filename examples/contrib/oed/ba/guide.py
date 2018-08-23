import torch
from torch import nn

import pyro
import pyro.distributions as dist

from pyro.contrib.oed.util import rmv


class Ba_lm_guide(nn.Module):

    def __init__(self, regu_shape, scale_tril_shape, w_sizes):
        super(Ba_lm_guide, self).__init__()
        self.regu = nn.Parameter(-2.*torch.ones(regu_shape))
        self.scale_tril = nn.Parameter(10.*torch.ones(scale_tril_shape))
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()

    def forward(self, y, design, target_label):

        # TODO fix this
        design = design[..., :self.w_sizes[target_label]]

        anneal = torch.diag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = tensorized_matrix_inverse(xtx)
        mu = torch.matmul(xtxi, torch.matmul(design.transpose(-1, -2), y.unsqueeze(-1))).squeeze(-1)

        scale_tril = tensorized_tril(self.scale_tril)
        print(scale_tril)

        return mu, scale_tril

    def guide(self, y_dict, design, observation_labels, target_labels):

        target_label = target_labels[0]
        pyro.module("ba_guide", self)

        y = y_dict["y"]
        mu, scale_tril = self.forward(y, design, target_label)

        # guide distributions for w
        w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        w = pyro.sample(target_label, w_dist)


class Ba_sigmoid_guide(nn.Module):

    def __init__(self, prior_sds, d, n, w_sizes):
        super(Ba_sigmoid_guide, self).__init__()
        self.anneal = torch.diag(prior_sds**2)
        p = prior_sds.shape[-1]
        self.inverse_sigmoid_scale = nn.Parameter(torch.ones(n))
        # self.inverse_sigmoid_offset = nn.Parameter(torch.zeros(n))
        self.scale_tril = nn.Parameter(10.*torch.ones(d, tri_n(p)))
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()
        self.regu = nn.Parameter(-2.*torch.ones(d, p))

    def forward(self, y, design, target_label):

        # Actual hard part, try to invert transformation on y
        # partial_logit = 1./y - 1.
        # logit = (partial_logit.clamp(1e-25, 1e25)).log()
        # y_trans = self.inverse_sigmoid_offset + logit/self.inverse_sigmoid_scale
        y, y1m = y.clamp(1e-35, 1), (1.-y).clamp(1e-35, 1)
        logited = y.log() - y1m.log()
        y_trans = logited/.1
        y_trans = y_trans * self.inverse_sigmoid_scale

        # TODO fix this
        design = design[..., :self.w_sizes[target_label]]

        anneal = tensorized_diag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = tensorized_matrix_inverse(xtx)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y_trans))

        scale_tril = tensorized_tril(self.scale_tril)

        return mu, scale_tril

    def guide(self, y_dict, design, observation_labels, target_labels):

        target_label = target_labels[0]
        pyro.module("ba_guide", self)

        y = y_dict["y"]
        mu, scale_tril = self.forward(y, design, target_label)

        # guide distributions for w
        w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        w = pyro.sample(target_label, w_dist)


class Ba_nig_guide(nn.Module):

    def __init__(self, regu_shape, scale_tril_shape, tau_shape, w_sizes, mf=False):
        super(Ba_nig_guide, self).__init__()
        self.regu = nn.Parameter(-2.*torch.ones(regu_shape))
        self.scale_tril = nn.Parameter(10.*torch.ones(scale_tril_shape))
        self.alpha = nn.Parameter(100.*torch.ones(tau_shape))
        self.b0 = nn.Parameter(100.*torch.ones(tau_shape))
        self.w_sizes = w_sizes
        self.mf = mf
        self.softplus = nn.Softplus()

    def forward(self, y, design, target_label):

        # TODO fix this
        design = design[..., :self.w_sizes[target_label]]

        anneal = torch.diag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = tensorized_matrix_inverse(xtx)
        mu = torch.matmul(xtxi, torch.matmul(design.transpose(-1, -2), y.unsqueeze(-1))).squeeze(-1)

        scale_tril = tensorized_tril(self.scale_tril)

        yty = torch.matmul(y.unsqueeze(-2), y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
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
        w = pyro.sample(target_label, w_dist)


def tensorized_matrix_inverse(M):
    if M.shape[-1] == 1:
        return 1./M
    elif M.shape[-1] == 2:
        det = M[..., 0, 0]*M[..., 1, 1] - M[..., 1, 0]*M[..., 0, 1]
        inv = torch.zeros(M.shape)
        inv[..., 0, 0] = M[..., 1, 1]
        inv[..., 1, 1] = M[..., 0, 0]
        inv[..., 0, 1] = -M[..., 0, 1]
        inv[..., 1, 0] = -M[..., 1, 0]
        inv = inv/det.unsqueeze(-1).unsqueeze(-1)
        return inv
    else:
        raise NotImplemented()


def tensorized_tril(M):
    if M.shape[-1] == 1:
        return M.unsqueeze(-1)
    if M.shape[-1] == 3:
        tril = torch.zeros(M.shape[:-1] + (2, 2))
        tril[..., 0, 0] = M[..., 0]
        tril[..., 1, 0] = M[..., 1]
        tril[..., 1, 1] = M[..., 2]
        return tril
    else:
        raise NotImplemented()


def tensorized_diag(M):
    if M.shape[-1] == 1:
        return M.unsqueeze(-1)
    if M.shape[-1] == 2:
        diag = torch.zeros(M.shape[:-1] + (2, 2))
        diag[..., 0, 0] = M[..., 0]
        diag[..., 1, 1] = M[..., 1]
        return diag
    else:
        raise NotImplemented()


def tri_n(n):
    return n*(n+1)/2
