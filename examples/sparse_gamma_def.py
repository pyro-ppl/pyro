# this is an implementation of the sparse gamma deep exponential family experiment presented in
# Ranganath, Rajesh, Tang, Linpeng, Charlin, Laurent, and Blei, David. Deep exponential families.
#
# the Olivetti faces dataset is originally from http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
#
# compare to Christian Naesseth's implementation here:
# https://github.com/blei-lab/ars-reparameterization/tree/master/sparse%20gamma%20def


from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
import torch

import pyro
import pyro.optim as optim
import pyro.poutine as poutine
import wget
from pyro.distributions import Gamma, Poisson
from pyro.infer import SVI, Trace_ELBO

torch.set_default_tensor_type('torch.FloatTensor')
pyro.enable_validation(True)
pyro.util.set_rng_seed(0)


class SparseGammaDEF(object):
    def __init__(self):
        # define the sizes of the layers in the deep exponential family
        self.top_width = 100
        self.mid_width = 40
        self.bottom_width = 15
        self.image_size = 64 * 64
        # define hyperpaameters that control the prior
        self.alpha_z = torch.tensor(0.1)
        self.beta_z = torch.tensor(0.1)
        self.alpha_w = torch.tensor(0.1)
        self.beta_w = torch.tensor(0.3)
        # define parameters used to initialize variational parameters
        self.alpha_init = 0.5
        self.mean_init = 0.0
        self.sigma_init = 0.1
        self.softplus = torch.nn.Softplus()

    # define the model
    @poutine.broadcast
    def model(self, x):
        x_size = x.size(0)

        # sample the global weights
        with pyro.iarange("w_top_iarange", self.top_width * self.mid_width):
            w_top = pyro.sample("w_top", Gamma(self.alpha_w, self.beta_w))
        with pyro.iarange("w_mid_iarange", self.mid_width * self.bottom_width):
            w_mid = pyro.sample("w_mid", Gamma(self.alpha_w, self.beta_w))
        with pyro.iarange("w_bottom_iarange", self.bottom_width * self.image_size):
            w_bottom = pyro.sample("w_bottom", Gamma(self.alpha_w, self.beta_w))

        # sample the local latent random variables
        # (the iarange encodes the fact that the z's for different datapoints are conditionally independent)
        with pyro.iarange("data", x_size):
            z_top = pyro.sample("z_top", Gamma(self.alpha_z, self.beta_z).expand([self.top_width]).independent(1))
            mean_mid = torch.mm(z_top, w_top.reshape(self.top_width, self.mid_width))
            z_mid = pyro.sample("z_mid", Gamma(self.alpha_z, self.beta_z / mean_mid).independent(1))
            mean_bottom = torch.mm(z_mid, w_mid.view(self.mid_width, self.bottom_width))
            z_bottom = pyro.sample("z_bottom", Gamma(self.alpha_z, self.beta_z / mean_bottom).independent(1))
            mean_obs = torch.mm(z_bottom, w_bottom.view(self.bottom_width, self.image_size))

            # observe the data using a poisson likelihood
            pyro.sample('obs', Poisson(mean_obs).independent(1), obs=x)

    # define the guide a.k.a. variational distribution.
    # (note the guide is mean field)
    @poutine.broadcast
    def guide(self, x):
        x_size = x.size(0)

        # helper for initializing variational parameters
        def rand_tensor(shape, mean, sigma):
            return mean * torch.ones(shape) + sigma * torch.randn(shape)

        # define a helper function to sample z's for a single layer
        def sample_zs(name, width):
            alpha_z_q = pyro.param("log_alpha_z_q_%s" % name,
                                   lambda: rand_tensor((x_size, width), self.alpha_init, self.sigma_init))
            mean_z_q = pyro.param("log_mean_z_q_%s" % name,
                                  lambda: rand_tensor((x_size, width), self.mean_init, self.sigma_init))
            alpha_z_q, mean_z_q = self.softplus(alpha_z_q), self.softplus(mean_z_q)
            pyro.sample("z_%s" % name, Gamma(alpha_z_q, alpha_z_q / mean_z_q).independent(1))

        # define a helper function to sample w's for a single layer
        def sample_ws(name, width):
            alpha_w_q = pyro.param("log_alpha_w_q_%s" % name,
                                   lambda: rand_tensor((width), self.alpha_init, self.sigma_init))
            mean_w_q = pyro.param("log_mean_w_q_%s" % name,
                                  lambda: rand_tensor((width), self.mean_init, self.sigma_init))
            alpha_w_q, mean_w_q = self.softplus(alpha_w_q), self.softplus(mean_w_q)
            pyro.sample("w_%s" % name, Gamma(alpha_w_q, alpha_w_q / mean_w_q))

        # sample the global weights
        with pyro.iarange("w_top_iarange", self.top_width * self.mid_width):
            sample_ws("top", self.top_width * self.mid_width)
        with pyro.iarange("w_mid_iarange", self.mid_width * self.bottom_width):
            sample_ws("mid", self.mid_width * self.bottom_width)
        with pyro.iarange("w_bottom_iarange", self.bottom_width * self.image_size):
            sample_ws("bottom", self.bottom_width * self.image_size)

        # sample the local latent random variables
        with pyro.iarange("data", x_size):
            sample_zs("top", self.top_width)
            sample_zs("mid", self.mid_width)
            sample_zs("bottom", self.bottom_width)

    # define a helper function to clip parameters defining the variational family.
    # (this is to avoid regions of the gamma distributions with extremely small means)
    def clip_params(self):
        for param, clip in zip(("log_alpha", "log_mean"), (-2.5, -4.5)):
            for layer in ["top", "mid", "bottom"]:
                for wz in ["_w_q_", "_z_q_"]:
                    pyro.param(param + wz + layer).data.clamp_(min=clip)


def main(args):
    # load data
    print('loading training data...')
    if not os.path.exists('faces_training.csv'):
        wget.download('https://d2fefpcigoriu7.cloudfront.net/datasets/faces_training.csv', 'faces_training.csv')
    data = torch.tensor(np.loadtxt('faces_training.csv', delimiter=',')).float()

    sparse_gamma_def = SparseGammaDEF()
    opt = optim.AdagradRMSProp({"eta": 4.5, "t": 0.1})
    svi = SVI(sparse_gamma_def.model, sparse_gamma_def.guide, opt, loss=Trace_ELBO())

    print('\nbeginning training...')

    # the training loop
    for k in range(args.num_epochs):
        loss = svi.step(data)
        sparse_gamma_def.clip_params()  # we clip params after each gradient step

        if k % 20 == 0 and k > 0:
            print("[epoch %04d] training elbo: %.4g" % (k, -loss))


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1001, type=int, help='number of training epochs')
    args = parser.parse_args()
    main(args)
