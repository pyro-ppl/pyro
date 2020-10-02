# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softplus, softmax

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO, Trace_ELBO

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from data import get_data


def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])


def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


def broadcast_inputs(input_args):
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args


# used in parameterizing p(z2 | z1, y)
class Z2Decoder(nn.Module):
    def __init__(self, z1_dim, y_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [z1_dim + y_dim] + hidden_dims + [2 * z2_dim]
        self.fc = make_fc(dims)

    def forward(self, z1, y):
        z1_y = torch.cat([z1, y], dim=-1)
        _z1_y = z1_y.reshape(-1, z1_y.size(-1))
        hidden = self.fc(_z1_y)
        hidden = hidden.reshape(z1_y.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale / 20.0)
        #scale = softplus(scale / 20.0 + 4.0)
        return loc, scale


# used in parameterizing p(x | z2)
class XDecoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_genes]
        self.fc = make_fc(dims)

    def forward(self, z2):
        mu = softmax(self.fc(z2), dim=-1)
        return mu


# used in parameterizing q(z2 | x) and q(l | x)
class Z2LEncoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [1 + num_genes] + hidden_dims + [2 * z2_dim + 2]
        self.fc = make_fc(dims)

    def forward(self, x, s):
        x_s = torch.cat([x, s], dim=-1)
        h1, h2 = split_in_half(self.fc(x_s))
        z2_loc, z2_scale = h1[..., :-1], softplus(h2[..., :-1] / 20.0 - 2.0)
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:] / 20.0 - 2.0)
        return z2_loc, z2_scale, l_loc, l_scale


# used in parameterizing q(z1 | z2, y)
class Z1Encoder(nn.Module):
    def __init__(self, num_labels, z1_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_labels + z2_dim] + hidden_dims + [2 * z1_dim]
        self.fc = make_fc(dims)

    def forward(self, z2, y):
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        _z2_y = z2_y.reshape(-1, z2_y.size(-1))
        hidden = self.fc(_z2_y)
        hidden = hidden.reshape(z2_y.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale / 20.0 - 2.0)
        return loc, scale


# used in parameterizing q(y | z2)
class Classifier(nn.Module):
    def __init__(self, z2_dim, hidden_dims, num_labels):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_labels]
        self.fc = make_fc(dims)

    def forward(self, x):
        logits = self.fc(x)
        return logits


class Spatial(nn.Module):
    def __init__(self, num_genes, num_labels, latent_dim=5, alpha=0.1, scale_factor=1.0):
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.scale_factor = scale_factor

        super().__init__()

        self.z2_decoder = Z2Decoder(z1_dim=self.latent_dim, y_dim=self.num_labels,
                                    z2_dim=self.latent_dim, hidden_dims=[50])
        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims=[100], z2_dim=self.latent_dim)
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[100])
        self.classifier = Classifier(z2_dim=self.latent_dim, hidden_dims=[50], num_labels=num_labels)
        self.z1_encoder = Z1Encoder(num_labels=num_labels, z1_dim=self.latent_dim,
                                    z2_dim=self.latent_dim, hidden_dims=[50])

        self.epsilon = 1.0e-6

    def model(self, l_mean, l_scale, x, s, y=None):
        pyro.module("spatial", self)

        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes),
                           constraint=constraints.positive)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample("z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            y = pyro.sample("y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)))

            z2_loc, z2_scale = self.z2_decoder(z1, y)
            #print("MODEL z2_loc", z2_loc.mean().item(), z2_loc.median().item())
            #print("MODEL z2_scale", z2_scale.mean().item(), z2_scale.median().item(), z2_scale.min().item(), z2_scale.max().item())
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            l_scale = l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(l_mean, l_scale).to_event(1))

            mu = self.x_decoder(z2)
            # TODO revisit this parameterization when https://github.com/pytorch/pytorch/issues/42449 is resolved
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            #print("theta", theta.mean().item(), "nb_logits", nb_logits.mean().item())
            x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits)
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, l_mean, l_scale, x, s, y=None):
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x, s)
            #print("z2_loc", z2_loc.mean().item(), "z2_scale",z2_scale.mean().item())
            pyro.sample("l", dist.LogNormal(l_loc + l_mean, l_scale).to_event(1))
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))
            #print("GUIDE z2_loc", z2_loc.mean().item(), z2_loc.median().item())
            #print("GUIDE z2_scale", z2_scale.mean().item(), z2_scale.median().item(), z2_scale.min().item(), z2_scale.max().item())
            #print("GUIDE z2", z2.mean().item(), z2.median().item(), z2.min().item(), z2.max().item())

            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                y = pyro.sample("y", y_dist)
            else:
                classification_loss = y_dist.log_prob(y)
                pyro.factor("classification_loss", -self.alpha * classification_loss)

            z1_loc, z1_scale = self.z1_encoder(z2, y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


def main(args):
    pyro.clear_param_store()
    pyro.util.set_rng_seed(args.seed)
    pyro.enable_validation(True)

    dataloader = get_data(mock=False, batch_size=args.batch_size)

    num_genes = dataloader.X_ref.size(-1)

    spatial = Spatial(num_genes=num_genes, num_labels=76,
                      scale_factor=1.0 / (args.batch_size * num_genes)).cuda()

    optim = ClippedAdam({"lr": args.learning_rate, "clip_norm": 10.0})
    guide = config_enumerate(spatial.guide, "parallel", expand=True)
    svi = SVI(spatial.model, guide, optim, TraceEnum_ELBO())

    for epoch in range(args.num_epochs):
        losses = []

        for x, yr, l_mean, l_scale, dataset in dataloader:
            if dataset == "ref":
                loss = svi.step(l_mean, l_scale, x, x.new_ones(x.size(0), 1), yr)
            elif dataset == "ss":
                loss = svi.step(l_mean, l_scale, x, x.new_zeros(x.size(0), 1), None)
            losses.append(loss)

        print("[Epoch %04d]  Loss: %.4f" % (epoch, np.mean(losses)))


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-s', '--seed', default=0, type=int, help='rng seed')
    parser.add_argument('-n', '--num-epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('-bs', '--batch-size', default=1024, type=int, help='mini-batch size')
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, help='learning rate')
    args = parser.parse_args()

    main(args)
