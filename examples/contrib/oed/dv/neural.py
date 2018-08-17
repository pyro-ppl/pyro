import torch
from torch import nn

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

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
    def __init__(self):
        super(T_specialized, self).__init__()
        self.obs_sd = torch.tensor(1.)
        self.w_sizes = (2,)
        self.regu = nn.Parameter(torch.tensor([10., 10.]))
        self.sds = nn.Parameter(torch.tensor([[10., 10.], [10., 10.]]))

    def forward(self, design, trace, observation_label, target_label="w"):
        trace.compute_log_prob()
        prior_lp = trace.nodes[target_label]["log_prob"]
        y = trace.nodes[observation_label]["value"]
        theta_dict = {target_label: trace.nodes[target_label]["value"]}

        design_suff = design.sum(-2, keepdim=True)
        suff = torch.matmul(y.unsqueeze(-2), design)
        mu = (suff/(design_suff + self.regu)).squeeze(-2)
        sigma = self.sds

        conditional_guide = pyro.condition(self.guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(design, mu, sigma)
        cond_trace.compute_log_prob()

        posterior_lp = cond_trace.nodes[target_label]["log_prob"]

        return posterior_lp - prior_lp

    def guide(self, design, mu, sigma):

        # pyro.module("ba_guide", self)

        # design is size batch x n x p
        # tau is size batch
        tau_shape = design.shape[:-2]

        # response will be shape batch x n
        obs_sd = self.obs_sd.expand(tau_shape).unsqueeze(-1)

        w_shape = tau_shape + self.w_sizes
        
        # guide distributions for w
        w_dist = dist.Normal(mu, sigma).independent(1)
        w = pyro.sample("w", w_dist)
