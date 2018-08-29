import torch
from torch import nn

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.contrib.oed.util import rmv, rinverse, rdiag


class T_neural(nn.Module):
    def __init__(self, design_dim, y_dim):
        super(T_neural, self).__init__()
        input_dim = design_dim + 2*y_dim + 1
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, design, trace, observation_label):
        trace.compute_log_prob()
        lp = trace.nodes[observation_label]["log_prob"]
        y = trace.nodes[observation_label]["value"]

        design_suff = design.sum(-2, keepdim=True)
        suff = torch.matmul(y.unsqueeze(-2), design)
        squares = torch.matmul((y**2).unsqueeze(-2), design)
        allsquare = suff**2
        lp_unsqueezed = lp.unsqueeze(-1).unsqueeze(-2)
        m = torch.cat([squares, allsquare, design_suff, lp_unsqueezed], -1)
        h1 = self.softplus(self.linear1(m))
        o = self.linear2(h1).squeeze(-2).squeeze(-1)
        return o


class T_specialized(nn.Module):
    def __init__(self, shape):
        super(T_specialized, self).__init__()
        self.regu = nn.Parameter(-2.*torch.ones(shape[-1] - 1))
        self.scale_tril = nn.Parameter(3.*torch.ones(shape))
        self.softplus = nn.Softplus()

    def forward(self, design, trace, observation_labels, target_labels):
        # TODO fix this
        observation_label = observation_labels[0]
        target_label = target_labels[0]

        trace.compute_log_prob()
        prior_lp = trace.nodes[target_label]["log_prob"]
        y = trace.nodes[observation_label]["value"]
        theta_dict = {target_label: trace.nodes[target_label]["value"]}

        anneal = torch.diag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = rinverse(xtx)
        mu = torch.matmul(xtxi, torch.matmul(design.transpose(-1, -2), y.unsqueeze(-1))).squeeze(-1)

        scale_tril = tensorized_tril(self.scale_tril)

        conditional_guide = pyro.condition(self.guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(design, mu, scale_tril, target_label)
        cond_trace.compute_log_prob()

        posterior_lp = cond_trace.nodes[target_label]["log_prob"]

        return posterior_lp - prior_lp

    def guide(self, design, mu, scale_tril, target_label):
        
        # guide distributions for w
        w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        pyro.sample(target_label, w_dist)


class T_sigmoid(nn.Module):
    def __init__(self, prior_sds, d, n, w_sizes):
        super(T_sigmoid, self).__init__()
        p = prior_sds.shape[-1]
        self.inverse_sigmoid_scale = nn.Parameter(torch.ones(n))
        self.h1_weight = nn.Parameter(torch.ones(n))
        self.h1_bias = nn.Parameter(torch.zeros(n))
        self.scale_tril = nn.Parameter(10.*torch.ones(d, tri_n(p)))
        self.regu = nn.Parameter(-2.*torch.ones(d, p))
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()

    def forward(self, design, trace, observation_labels, target_labels):
        # TODO fix this
        observation_label = observation_labels[0]
        target_label = target_labels[0]

        trace.compute_log_prob()
        prior_lp = trace.nodes[target_label]["log_prob"]
        y = trace.nodes[observation_label]["value"]
        theta_dict = {target_label: trace.nodes[target_label]["value"]}

        # Approx invert transformation on y in expectation
        y, y1m = y.clamp(1e-35, 1), (1.-y).clamp(1e-35, 1)
        logited = y.log() - y1m.log()
        y_trans = logited/.1
        y_trans = y_trans * self.inverse_sigmoid_scale
        hidden = self.softplus(y_trans)
        y_trans = y_trans + self.h1_bias

        # TODO fix this
        design = design[..., :self.w_sizes[target_label]]

        anneal = rdiag(self.softplus(self.regu))
        xtx = torch.matmul(design.transpose(-1, -2), design) + anneal
        xtxi = rinverse(xtx)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y_trans))

        scale_tril = tensorized_tril(self.scale_tril)

        # return mu, scale_tril
        conditional_guide = pyro.condition(self.guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(design, mu, scale_tril, target_labels)
        cond_trace.compute_log_prob()

        posterior_lp = cond_trace.nodes[target_label]["log_prob"]

        return posterior_lp - prior_lp

    def guide(self, design, mu, scale_tril, target_labels):

        target_label = target_labels[0]

        # guide distributions for w
        w_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
        pyro.sample(target_label, w_dist)


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


def tri_n(n):
    return n*(n+1)/2
