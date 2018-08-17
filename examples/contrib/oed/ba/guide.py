import torch
from torch import nn

import pyro
import pyro.distributions as dist


class Ba_lm_guide(nn.Module):

    def __init__(self, w_sds, obs_sd=torch.tensor(1.)):
        super(Ba_lm_guide, self).__init__()
        self.obs_sd = obs_sd
        self.w_sizes = w_sds.shape
        self.w_sds = w_sds
        self.regu = nn.Parameter(torch.tensor([10., 10.]))
        self.sds = nn.Parameter(torch.tensor([[10., 10.], [10., 10.]]))

    def forward(self, y, design):
        prior_var = self.w_sds**2

        design_suff = design.sum(-2, keepdim=True)
        suff = torch.matmul(y.unsqueeze(-2), design)
        mu = (suff/(design_suff + self.regu)).squeeze(-2)
        mu2 = (suff/(design_suff + 1./prior_var)).squeeze(-2)
        print('analytic mu', mu2)
        print('trained mu', mu)
        print('analytic regu', 1./prior_var)
        print('trained regu', self.regu)

        posterior_var = prior_var - (1./(prior_var*design_suff + self.obs_sd**2))*(prior_var**2*design_suff)
        posterior_var = posterior_var.squeeze(-2)
        print('analytic sds', posterior_var.sqrt())
        print('trained sds', self.sds)
        return mu, self.sds

    def guide(self, y_dict, design):

        pyro.module("ba_guide", self)

        y = y_dict["y"]
        # design is size batch x n x p
        # tau is size batch
        tau_shape = design.shape[:-2]

        # response will be shape batch x n
        obs_sd = self.obs_sd.expand(tau_shape).unsqueeze(-1)

        w_shape = tau_shape + self.w_sizes
        # Set up mu and lambda
        mu, sigma = self.forward(y, design)
        # print('y', y)
        # print('mu', mu)
        # print('sigma', sigma)
        
        # guide distributions for w
        w_dist = dist.Normal(mu, sigma).independent(1)
        w = pyro.sample("w", w_dist)
