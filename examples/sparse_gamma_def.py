# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

# This is an implementation of the sparse gamma deep exponential family model described in
# Ranganath, Rajesh, Tang, Linpeng, Charlin, Laurent, and Blei, David. Deep exponential families.
#
# To do inference we use one of the following guides:
# i)   a custom guide (i.e. a hand-designed variational family) or
# ii)  an 'auto' guide that is automatically constructed using pyro.infer.autoguide or
# iii) an 'easy' guide whose construction is facilitated using pyro.contrib.easyguide.
#
# The Olivetti faces dataset is originally from http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
#
# Compare to Christian Naesseth's implementation here:
# https://github.com/blei-lab/ars-reparameterization/tree/master/sparse%20gamma%20def

import argparse
import errno
import os

import numpy as np
import torch
from torch.nn.functional import softplus

import pyro
import pyro.optim as optim
import wget

from pyro.contrib.examples.util import get_data_directory
from pyro.distributions import Gamma, Poisson, Normal
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide import init_to_feasible
from pyro.contrib.easyguide import EasyGuide


torch.set_default_tensor_type('torch.FloatTensor')
pyro.enable_validation(__debug__)
pyro.util.set_rng_seed(0)


# helper for initializing variational parameters
def rand_tensor(shape, mean, sigma):
    return mean * torch.ones(shape) + sigma * torch.randn(shape)


class SparseGammaDEF:
    def __init__(self):
        # define the sizes of the layers in the deep exponential family
        self.top_width = 100
        self.mid_width = 40
        self.bottom_width = 15
        self.image_size = 64 * 64
        # define hyperparameters that control the prior
        self.alpha_z = torch.tensor(0.1)
        self.beta_z = torch.tensor(0.1)
        self.alpha_w = torch.tensor(0.1)
        self.beta_w = torch.tensor(0.3)
        # define parameters used to initialize variational parameters
        self.alpha_init = 0.5
        self.mean_init = 0.0
        self.sigma_init = 0.1

    # define the model
    def model(self, x):
        x_size = x.size(0)

        # sample the global weights
        with pyro.plate("w_top_plate", self.top_width * self.mid_width):
            w_top = pyro.sample("w_top", Gamma(self.alpha_w, self.beta_w))
        with pyro.plate("w_mid_plate", self.mid_width * self.bottom_width):
            w_mid = pyro.sample("w_mid", Gamma(self.alpha_w, self.beta_w))
        with pyro.plate("w_bottom_plate", self.bottom_width * self.image_size):
            w_bottom = pyro.sample("w_bottom", Gamma(self.alpha_w, self.beta_w))

        # sample the local latent random variables
        # (the plate encodes the fact that the z's for different datapoints are conditionally independent)
        with pyro.plate("data", x_size):
            z_top = pyro.sample("z_top", Gamma(self.alpha_z, self.beta_z).expand([self.top_width]).to_event(1))
            # note that we need to use matmul (batch matrix multiplication) as well as appropriate reshaping
            # to make sure our code is fully vectorized
            w_top = w_top.reshape(self.top_width, self.mid_width) if w_top.dim() == 1 else \
                w_top.reshape(-1, self.top_width, self.mid_width)
            mean_mid = torch.matmul(z_top, w_top)
            z_mid = pyro.sample("z_mid", Gamma(self.alpha_z, self.beta_z / mean_mid).to_event(1))

            w_mid = w_mid.reshape(self.mid_width, self.bottom_width) if w_mid.dim() == 1 else \
                w_mid.reshape(-1, self.mid_width, self.bottom_width)
            mean_bottom = torch.matmul(z_mid, w_mid)
            z_bottom = pyro.sample("z_bottom", Gamma(self.alpha_z, self.beta_z / mean_bottom).to_event(1))

            w_bottom = w_bottom.reshape(self.bottom_width, self.image_size) if w_bottom.dim() == 1 else \
                w_bottom.reshape(-1, self.bottom_width, self.image_size)
            mean_obs = torch.matmul(z_bottom, w_bottom)

            # observe the data using a poisson likelihood
            pyro.sample('obs', Poisson(mean_obs).to_event(1), obs=x)

    # define our custom guide a.k.a. variational distribution.
    # (note the guide is mean field gamma)
    def guide(self, x):
        x_size = x.size(0)

        # define a helper function to sample z's for a single layer
        def sample_zs(name, width):
            alpha_z_q = pyro.param("alpha_z_q_%s" % name,
                                   lambda: rand_tensor((x_size, width), self.alpha_init, self.sigma_init))
            mean_z_q = pyro.param("mean_z_q_%s" % name,
                                  lambda: rand_tensor((x_size, width), self.mean_init, self.sigma_init))
            alpha_z_q, mean_z_q = softplus(alpha_z_q), softplus(mean_z_q)
            pyro.sample("z_%s" % name, Gamma(alpha_z_q, alpha_z_q / mean_z_q).to_event(1))

        # define a helper function to sample w's for a single layer
        def sample_ws(name, width):
            alpha_w_q = pyro.param("alpha_w_q_%s" % name,
                                   lambda: rand_tensor((width), self.alpha_init, self.sigma_init))
            mean_w_q = pyro.param("mean_w_q_%s" % name,
                                  lambda: rand_tensor((width), self.mean_init, self.sigma_init))
            alpha_w_q, mean_w_q = softplus(alpha_w_q), softplus(mean_w_q)
            pyro.sample("w_%s" % name, Gamma(alpha_w_q, alpha_w_q / mean_w_q))

        # sample the global weights
        with pyro.plate("w_top_plate", self.top_width * self.mid_width):
            sample_ws("top", self.top_width * self.mid_width)
        with pyro.plate("w_mid_plate", self.mid_width * self.bottom_width):
            sample_ws("mid", self.mid_width * self.bottom_width)
        with pyro.plate("w_bottom_plate", self.bottom_width * self.image_size):
            sample_ws("bottom", self.bottom_width * self.image_size)

        # sample the local latent random variables
        with pyro.plate("data", x_size):
            sample_zs("top", self.top_width)
            sample_zs("mid", self.mid_width)
            sample_zs("bottom", self.bottom_width)


# define a helper function to clip parameters defining the custom guide.
# (this is to avoid regions of the gamma distributions with extremely small means)
def clip_params():
    for param, clip in zip(("alpha", "mean"), (-2.5, -4.5)):
        for layer in ["_q_top", "_q_mid", "_q_bottom"]:
            for wz in ["_w", "_z"]:
                pyro.param(param + wz + layer).data.clamp_(min=clip)


# Define a guide using the EasyGuide class.
# Unlike the 'auto' guide, this guide supports data subsampling.
# This is the best performing of the three guides.
#
# This guide is functionally similar to the auto guide, but performs
# somewhat better. The reason seems to be some combination of: i) the better
# numerical stability of the softplus; and ii) the custom initialization.
# Note however that for both the easy guide and auto guide KL divergences
# are not computed analytically in the ELBO because the ELBO thinks the
# mean-field condition is not satisfied, which leads to higher variance gradients.
class MyEasyGuide(EasyGuide):
    def guide(self, x):
        # group all the latent weights into one large latent variable
        global_group = self.group(match="w_.*")
        global_mean = pyro.param("w_mean",
                                 lambda: rand_tensor(global_group.event_shape, 0.5, 0.1))
        global_scale = softplus(pyro.param("w_scale",
                                lambda: rand_tensor(global_group.event_shape, 0.0, 0.1)))
        # use a mean field Normal distribution on all the ws
        global_group.sample("ws", Normal(global_mean, global_scale).to_event(1))

        # group all the latent zs into one large latent variable
        local_group = self.group(match="z_.*")
        x_shape = x.shape[:1] + local_group.event_shape

        with self.plate("data", x.size(0)):
            local_mean = pyro.param("z_mean",
                                    lambda: rand_tensor(x_shape, 0.5, 0.1))
            local_scale = softplus(pyro.param("z_scale",
                                   lambda: rand_tensor(x_shape, 0.0, 0.1)))
            # use a mean field Normal distribution on all the zs
            local_group.sample("zs", Normal(local_mean, local_scale).to_event(1))


def main(args):
    # load data
    print('loading training data...')
    dataset_directory = get_data_directory(__file__)
    dataset_path = os.path.join(dataset_directory, 'faces_training.csv')
    if not os.path.exists(dataset_path):
        try:
            os.makedirs(dataset_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass
        wget.download('https://d2hg8soec8ck9v.cloudfront.net/datasets/faces_training.csv', dataset_path)
    data = torch.tensor(np.loadtxt(dataset_path, delimiter=',')).float()

    sparse_gamma_def = SparseGammaDEF()

    # Due to the special logic in the custom guide (e.g. parameter clipping), the custom guide
    # seems to be more amenable to higher learning rates.
    # Nevertheless, the easy guide performs the best (presumably because of numerical instabilities
    # related to the gamma distribution in the custom guide).
    learning_rate = 0.2 if args.guide in ['auto', 'easy'] else 4.5
    momentum = 0.05 if args.guide in ['auto', 'easy'] else 0.1
    opt = optim.AdagradRMSProp({"eta": learning_rate, "t": momentum})

    # use one of our three different guide types
    if args.guide == 'auto':
        guide = AutoDiagonalNormal(sparse_gamma_def.model, init_loc_fn=init_to_feasible)
    elif args.guide == 'easy':
        guide = MyEasyGuide(sparse_gamma_def.model)
    else:
        guide = sparse_gamma_def.guide

    # this is the svi object we use during training; we use TraceMeanField_ELBO to
    # get analytic KL divergences
    svi = SVI(sparse_gamma_def.model, guide, opt, loss=TraceMeanField_ELBO())

    # we use svi_eval during evaluation; since we took care to write down our model in
    # a fully vectorized way, this computation can be done efficiently with large tensor ops
    svi_eval = SVI(sparse_gamma_def.model, guide, opt,
                   loss=TraceMeanField_ELBO(num_particles=args.eval_particles, vectorize_particles=True))

    print('\nbeginning training with %s guide...' % args.guide)

    # the training loop
    for k in range(args.num_epochs):
        loss = svi.step(data)
        # for the custom guide we clip parameters after each gradient step
        if args.guide == 'custom':
            clip_params()

        if k % args.eval_frequency == 0 and k > 0 or k == args.num_epochs - 1:
            loss = svi_eval.evaluate_loss(data)
            print("[epoch %04d] training elbo: %.4g" % (k, -loss))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1500, type=int, help='number of training epochs')
    parser.add_argument('-ef', '--eval-frequency', default=25, type=int,
                        help='how often to evaluate elbo (number of epochs)')
    parser.add_argument('-ep', '--eval-particles', default=20, type=int,
                        help='number of samples/particles to use during evaluation')
    parser.add_argument('--guide', default='custom', type=str,
                        help='use a custom, auto, or easy guide')
    args = parser.parse_args()
    assert args.guide in ['custom', 'auto', 'easy']
    main(args)
