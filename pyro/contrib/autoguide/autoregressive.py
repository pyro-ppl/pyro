from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import biject_to, constraints
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoContinuous

class SimpleEncoder(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim):
                super(SimpleEncoder, self).__init__()
                # setup the three linear transformations used
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc21 = nn.Linear(hidden_dim, output_dim)
                self.fc22 = nn.Linear(hidden_dim, output_dim)
                # setup the non-linearities
                self.softplus = nn.ReLU()

            def forward(self, x):
                # define the forward computation on the image x
                x = x.reshape(-1,1).squeeze(0)
                # then compute the hidden units
                hidden = self.softplus(self.fc1(x))
                # then return a mean vector and a log(sqrt(covar))
                # each of size batch_size x z_dim
                z_loc = self.fc21(hidden)
                z_lg_scale = self.fc22(hidden)

                return z_loc, z_lg_scale

class AutoAutoregressiveNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Cholesky
    factorization of a Multivariate Normal distribution to construct a guide
    over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.

    Usage::

        guide = AutoMultivariateNormal(model)
        svi = SVI(model, guide, ...)

    By default the mean vector is initialized to zero and the Cholesky factor
    is initialized to the identity.  To change this default behavior the user
    should call :func:`pyro.param` before beginning inference, e.g.::

        latent_dim = 10
        pyro.param("auto_loc", torch.randn(latent_dim))
        pyro.param("auto_scale_tril", torch.tril(torch.rand(latent_dim)),
                   constraint=constraints.lower_cholesky)
    """
    def __init__(self, *args, **kwargs):
        super(AutoAutoregressiveNormal, self).__init__(*args, **kwargs)

        # Create the NNs to regress the values of each variable's parents to its params
        # NOTE: The 0th variable has no parents, hence no NN
        self.nns = nn.ModuleList()
        for idx in range(1, self.latent_dim):
            self.nns.append(SimpleEncoder(input_dim=(idx), output_dim=1, hidden_dim=16))
            pyro.module(f"{self.prefix}_{idx}_encoder", self.nns[-1])
        
    
    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        unconstrained_samples = []*self.latent_dim

        # Sample the first variable that doesn't use a NN
        z0_loc = pyro.param('z0_loc', torch.tensor(0.))
        z0_lg_scale = pyro.param('z1_lg_scale', torch.tensor(0.0))
        unconstrained_samples[0] = pyro.sample('z0', dist.Normal(z0_loc, torch.exp(z0_lg_scale)), infer={'is_auxiliary': True})

        # Sample the rest that do
        for idx in range(1, self.latent_dim):
            z_prev = torch.stack(unconstrained_samples[:idx])
            z_loc, z_lg_scale = self.nns[idx-1](z_prev)
            unconstrained_samples[idx] = pyro.sample(f'z{idx}', dist.Normal(z_loc, torch.exp(z_lg_scale)), infer={'is_auxiliary': True})

        pos_dist = dist.Delta(v=torch.stack(unconstrained_samples))
        return pyro.sample("_{}_latent".format(self.prefix), pos_dist, infer={"is_auxiliary": True})