"""
We use a semi-supervised deep generative model of RNAseq data to propagate labels
from a small set of labeled cells to a larger set of unlabeled cells.

References:
[1] "Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models,"
    Chenling Xu, Romain Lopez, Edouard Mehlman, Jeffrey Regier, Michael I. Jordan, Nir Yosef.
[2] https://github.com/YosefLab/scvi-tutorials/blob/50dd3269abfe0c375ec47114f2c20725a016736f/seed_labeling.ipynb
"""

import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softplus, softmax

import pyro
import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.optim import ClippedAdam
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO

from data import get_data_loader


# helper for making fully-connected neural networks
def make_fc(dims, use_batchnorm=True):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])  # exclude final ReLU non-linearity


# splits a tensor in half along the final dimension
def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


# helper for broadcasting inputs to neural net
def broadcast_inputs(input_args):
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args


# used in parameterizing p(z2 | z1, y)
class Z2Decoder(nn.Module):
    def __init__(self, z1_dim, y_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [z1_dim + y_dim] + hidden_dims + [2 * z2_dim]
        self.fc = make_fc(dims, use_batchnorm=False)

    def forward(self, z1, y):
        z1_y = torch.cat([z1, y], dim=-1)
        loc, scale = split_in_half(self.fc(z1_y))
        scale = softplus(scale)
        return loc, scale


# used in parameterizing p(x | z2)
class XDecoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)

    def forward(self, z2):
        gate, mu = split_in_half(self.fc(z2))
        gate = gate.sigmoid()
        mu = softmax(mu, dim=-1)
        return gate, mu


# used in parameterizing q(z2 | x) and q(l | x)
class Z2LEncoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2 * z2_dim + 2]
        self.fc = make_fc(dims)

    def forward(self, x):
        h1, h2 = split_in_half(self.fc(x))
        z2_loc, z2_scale = h1[..., :-1], softplus(h2[..., :-1])
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:])
        l_loc = l_loc.clamp(max=10.0)
        l_scale = l_scale.clamp(max=3.0)
        return z2_loc, z2_scale, l_loc, l_scale


# used in parameterizing q(z1 | z2, y)
class Z1Encoder(nn.Module):
    def __init__(self, num_labels, z1_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_labels + z2_dim] + hidden_dims + [2 * z1_dim]
        self.fc = make_fc(dims, use_batchnorm=False)

    def forward(self, z2, y):
        # this is necessary since Pyro expands y during enumeration
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        loc, scale = split_in_half(self.fc(z2_y))
        scale = softplus(scale)
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


# encompasses the scANVI model and guide as a PyTorch nn.Module
class SCANVI(nn.Module):
    def __init__(self, num_genes, num_labels, l_loc=5.0, l_scale=5.0, alpha=0.1, scale_factor=1.0):
        assert isinstance(num_labels, int) and num_labels > 1
        self.num_labels = num_labels

        assert isinstance(num_genes, int)
        self.num_genes = num_genes

        self.latent_dim = 5

        assert isinstance(l_loc, float)
        self.l_loc = l_loc

        assert isinstance(l_scale, float) and l_scale > 0
        self.l_scale = l_scale

        assert isinstance(alpha, float) and alpha > 0
        self.alpha = alpha

        assert isinstance(scale_factor, float) and scale_factor > 0
        self.scale_factor = scale_factor

        super().__init__()

        self.z2_decoder = Z2Decoder(z1_dim=self.latent_dim, y_dim=self.num_labels,
                                    z2_dim=self.latent_dim, hidden_dims=[50])
        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims=[50], z2_dim=self.latent_dim)
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[50])
        self.classifier = Classifier(z2_dim=self.latent_dim, hidden_dims=[50], num_labels=num_labels)
        self.z1_encoder = Z1Encoder(num_labels=num_labels, z1_dim=self.latent_dim,
                                    z2_dim=self.latent_dim, hidden_dims=[50])

    def model(self, x, y=None):
        # register various nn.Modules with Pyro
        pyro.module("scanvi", self)

        theta = pyro.param("inverse_dispersion", x.new_ones(self.num_genes),
                           constraint=constraints.positive)

        with pyro.plate("batch", len(x)), pyro.poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample("z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            y = pyro.sample("y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)))

            z2_loc, z2_scale = self.z2_decoder(z1, y)
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            l_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(self.l_loc, l_scale).to_event(1))

            gate, mu = self.x_decoder(z2)
            eps = 1.0e-6
            nb_logits = (theta + eps).log() - (l * mu + eps).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(gate=gate, total_count=theta,
                                                       logits=nb_logits)
            x = pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, y=None):
        pyro.module("scanvi", self)
        with pyro.plate("batch", len(x)), pyro.poutine.scale(scale=self.scale_factor):
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                # x is unlabeled so sample y using q(y|z2)
                y = pyro.sample("y", y_dist)
            else:
                # x is labeled so add a classification loss term
                classification_loss = y_dist.log_prob(y)
                pyro.factor("classification_loss", -self.alpha * classification_loss)

            z1_loc, z1_scale = self.z1_encoder(z2, y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


def main(args):
    # clear param store
    pyro.clear_param_store()
    # fix random number seed
    pyro.util.set_rng_seed(args.seed)
    # enable optional validation warnings
    pyro.enable_validation(True)

    num_genes = 1292

    scanvi = SCANVI(num_genes=num_genes, num_labels=4, scale_factor=1.0 / (args.batch_size * num_genes))
    if args.cuda:
        scanvi.cuda()

    dataloader = get_data_loader(batch_size=args.batch_size, cuda=args.cuda)

    # setup an optimizer
    # we decay the learning rate over the course of learning
    lrd = 0.1 ** (1.0 / (len(dataloader) * args.num_epochs))
    optim = ClippedAdam({"lr": args.learning_rate, "clip_norm": 0.1, "lrd": lrd})

    # tell Pyro to enumerate out y when y is unobserved
    guide = config_enumerate(scanvi.guide, "parallel", expand=True)

    # setup a variational objective for gradient-based learning
    svi = SVI(scanvi.model, guide, optim, TraceEnum_ELBO())

    # training loop
    for epoch in range(args.num_epochs):
        losses = []

        for x, y in dataloader:
            loss = svi.step(x, y)
            losses.append(loss)

        print("[Epoch %04d]  Loss: %.4f" % (epoch, np.mean(losses)))


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-s', '--seed', default=0, type=int, help='rng seed')
    parser.add_argument('-n', '--num-epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('-bs', '--batch-size', default=100, type=int, help='mini-batch size')
    parser.add_argument('-lr', '--learning-rate', default=0.002, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    args = parser.parse_args()

    main(args)
