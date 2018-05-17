from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch import nn as nn

import pyro
import pyro.optim as optim
from pyro.distributions import Gamma, Poisson
from pyro.infer import SVI, Trace_ELBO


torch.set_default_tensor_type('torch.FloatTensor')
pyro.enable_validation(True)
pyro.util.set_rng_seed(0)
data = torch.tensor(np.loadtxt('faces_training.csv', delimiter=',')).float()


class SparseGammaDEF(object):
    def __init__(self):
        self.top_width = 100
        self.mid_width = 40
        self.bottom_width = 15
        self.image_size = 64 * 64
        self.alpha_z = torch.tensor(0.1)
        self.alpha_w = torch.tensor(0.1)
        self.beta_w = torch.tensor(0.3)
        self.alpha_init = 0.5
        self.mean_init = 0.0
        self.sigma_init = 0.1
        self.softplus = nn.Softplus()

    def model(self, x):
        x_size = x.size(0)
        with pyro.iarange('data'):
            with pyro.iarange('locals'):
                z_top = pyro.sample("z_top", Gamma(self.alpha_z.expand(x_size, self.top_width),
                                                   self.alpha_z.expand(x_size, self.top_width)).independent(1))
                w_top = pyro.sample("w_top", Gamma(self.alpha_w.expand(self.top_width * self.mid_width),
                                                   self.beta_w.expand(self.top_width * self.mid_width)).independent(1))
                mean_mid = torch.mm(z_top, w_top.view(self.top_width, self.mid_width))

                z_mid = pyro.sample("z_mid",
                                    Gamma(self.alpha_z.expand(x_size, self.mid_width),
                                          self.alpha_z.expand(x_size, self.mid_width) / mean_mid).independent(1))
                w_mid = pyro.sample("w_mid",
                                    Gamma(self.alpha_w.expand(self.mid_width * self.bottom_width),
                                          self.beta_w.expand(self.mid_width * self.bottom_width)).independent(1))
                mean_bottom = torch.mm(z_mid, w_mid.view(self.mid_width, self.bottom_width))

                z_bottom = pyro.sample("z_bottom",
                                       Gamma(self.alpha_z.expand(x_size, self.bottom_width),
                                             self.alpha_z.expand(x_size, self.bottom_width) / mean_bottom)
                                       .independent(1))
                w_bottom = pyro.sample("w_bottom",
                                       Gamma(self.alpha_w.expand(self.bottom_width * self.image_size),
                                             self.beta_w.expand(self.bottom_width * self.image_size)).independent(1))
                mean_obs = torch.mm(z_bottom, w_bottom.view(self.bottom_width, self.image_size))

            pyro.sample('obs', Poisson(mean_obs).independent(1), obs=x)

    def guide(self, x):
        x_size = x.size(0)

        # define a helper function to sample z's for a single layer
        def sample_zs(name, width):
            alpha_z_q = pyro.param("log_alpha_z_q_%s" % name,
                                   torch.tensor(self.alpha_init * torch.ones(x_size, width) +
                                                self.sigma_init * torch.randn(x_size, width)))
            mean_z_q = pyro.param("log_mean_z_q_%s" % name,
                                  torch.tensor(self.mean_init * torch.ones(x_size, width) +
                                               self.sigma_init * torch.randn(x_size, width)))

            alpha_z_q, mean_z_q = self.softplus(alpha_z_q), self.softplus(mean_z_q)
            pyro.sample("z_%s" % name, Gamma(alpha_z_q, alpha_z_q / mean_z_q).independent(1))

        # define a helper function to sample w's for a single layer
        def sample_ws(name, width):
            alpha_w_q = pyro.param("log_alpha_w_q_%s" % name,
                                   torch.tensor(self.alpha_init * torch.ones(width) +
                                                self.sigma_init * torch.randn(width)))
            mean_w_q = pyro.param("log_mean_w_q_%s" % name,
                                  torch.tensor(self.mean_init * torch.ones(width) +
                                               self.sigma_init * torch.randn(width)))
            alpha_w_q, mean_w_q = self.softplus(alpha_w_q), self.softplus(mean_w_q)
            pyro.sample("w_%s" % name, Gamma(alpha_w_q, alpha_w_q / mean_w_q).independent(1))

        # sample latent random variables.
        # note that we need to enclose everything in two pyro.iarange's to encapsulate the fact that
        # -- the latents for each datapoint are conditionally independent of the latents for other datapoints
        # -- the different dimensions of z_top etc. for a particular datapoint are conditionally
        #    independent of other dimensions
        with pyro.iarange('data'):
            with pyro.iarange('locals'):
                sample_zs("top", self.top_width)
                sample_zs("mid", self.mid_width)
                sample_zs("bottom", self.bottom_width)

                sample_ws("top", self.top_width * self.mid_width)
                sample_ws("mid", self.mid_width * self.bottom_width)
                sample_ws("bottom", self.bottom_width * self.image_size)

    # define a helper function to clip parameters defining the variational family
    def clip_params(self):
        for param, clip in zip(("log_alpha", "log_mean"), (-2.25, -4.5)):
            for layer in ["top", "mid", "bottom"]:
                for wz in ["_w_q_", "_z_q_"]:
                    pyro.param(param + wz + layer).data.clamp_(min=clip)


sparse_gamma_def = SparseGammaDEF()
opt = optim.AdagradRMSProp({"eta": 4.5, "t": 0.1})
svi = SVI(sparse_gamma_def.model, sparse_gamma_def.guide, opt, loss=Trace_ELBO())

for k in range(1000):
    loss = svi.step(data)
    sparse_gamma_def.clip_params()

    if k % 20 == 0 and k > 0:
        print("[epoch %05d] training elbo: %.4g" % (k, -loss))
