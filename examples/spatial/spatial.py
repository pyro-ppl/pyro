# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import numpy as np
from scipy.cluster.vq import kmeans2

import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softplus, softmax

import gpytorch
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO, Trace_ELBO

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import scanpy as sc

from data import get_data


def make_fc(dims, dropout=0.0):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        #if dropout > 0.0:
        #    layers.append(nn.Dropout(p=dropout))
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
        scale = softplus(scale / 2.0)
        return loc, scale


# used in parameterizing p(x | z2)
class XDecoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [z2_dim + 1] + hidden_dims + [num_genes]
        self.fc = make_fc(dims)

    def forward(self, z2, s):
        z2_s = torch.cat([z2, s], dim=-1)
        mu = softmax(self.fc(z2_s), dim=-1)
        return mu


# used in parameterizing q(z2 | x) and q(l | x)
class Z2LEncoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [1 + num_genes] + hidden_dims + [2 * z2_dim + 2]
        self.fc = make_fc(dims, dropout=0.0)

    def forward(self, x, s):
        x = torch.log(1.0 + x)
        x_s = torch.cat([x, s], dim=-1)
        h1, h2 = split_in_half(self.fc(x_s))
        z2_loc, z2_scale = h1[..., :-1], softplus(h2[..., :-1] / 2.0 - 1.0)
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:] / 2.0 - 1.0)
        return z2_loc, z2_scale, l_loc, l_scale


# used in parameterizing q(z1 | z2, y)
class Z1Encoder(nn.Module):
    def __init__(self, num_classes, z1_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_classes + z2_dim] + hidden_dims + [2 * z1_dim]
        self.fc = make_fc(dims, dropout=0.0)

    def forward(self, z2, y):
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        _z2_y = z2_y.reshape(-1, z2_y.size(-1))
        hidden = self.fc(_z2_y)
        hidden = hidden.reshape(z2_y.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale / 2.0 - 1.0)
        return loc, scale


# used in parameterizing q(y | z2)
class Classifier(nn.Module):
    def __init__(self, z2_dim, hidden_dims, num_classes):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_classes]
        self.fc = make_fc(dims)

    def forward(self, z2):
        logits = self.fc(z2)
        return logits


class SpatialGP(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing=100, num_classes=3, R_ss=None, beta=1.0, name_prefix="spatial_gp"):
        print("Initialized SpatialGP with num_inducing=%d, num_classes=%d, beta=%.3f" % (num_inducing,
               num_classes, beta))
        self.name_prefix = name_prefix
        self.num_classes = num_classes
        self.beta = beta

        inducing_points = R_ss.clone().data.cpu().numpy()[torch.randperm(R_ss.size(0))[:num_inducing]]
        inducing_points = torch.tensor(kmeans2(R_ss.data.cpu().numpy(),
                                               inducing_points, minit='matrix')[0]).cuda()
        inducing_points = inducing_points.expand(num_classes, num_inducing, 2).clone().detach()

        batch_shape = torch.Size([num_classes])
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing,
                                                                   batch_shape=batch_shape)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(MaternKernel(batch_shape=batch_shape, ard_num_dims=2),
                                                     batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    @torch.no_grad()
    def qf(self, x):
        return pyro.sample(self.name_prefix + ".f(x)", self.pyro_guide(x)).t()

    def guide(self, x):
        with pyro.plate("classes_plate", self.num_classes):
            return pyro.sample(self.name_prefix + ".f(x)", self.pyro_guide(x, beta=self.beta))

    def model(self, x):
        with pyro.plate("classes_plate", self.num_classes):
            return pyro.sample(self.name_prefix + ".f(x)", self.pyro_model(x, beta=self.beta))



class Spatial(nn.Module):
    def __init__(self, num_genes, num_classes, spatial_gp, latent_dim=10, alpha=0.01, scale_factor=1.0):
        self.num_genes = num_genes
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.scale_factor = scale_factor

        print("Initialized Spatial with num_genes=%d, num_classes=%d, latent_dim=%d" % (num_genes,
               num_classes, latent_dim))
        super().__init__()

        self.z2_decoder = Z2Decoder(z1_dim=self.latent_dim, y_dim=self.num_classes,
                                    z2_dim=self.latent_dim, hidden_dims=[50])
        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims=[100], z2_dim=self.latent_dim)
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[100])
        self.classifier = Classifier(z2_dim=self.latent_dim, hidden_dims=[50], num_classes=num_classes)
        self.z1_encoder = Z1Encoder(num_classes=num_classes, z1_dim=self.latent_dim,
                                    z2_dim=self.latent_dim, hidden_dims=[50])
        self.gp = spatial_gp

        self.epsilon = 1.0e-6

        pyro.param("inverse_dispersion_ref", 10.0 * torch.ones(self.num_genes).cuda(), constraint=constraints.positive)
        pyro.param("inverse_dispersion_ss", 10.0 * torch.ones(self.num_genes).cuda(), constraint=constraints.positive)

    def model(self, l_mean, l_scale, x, s, y=None, r=None, anneal=1.0):
        pyro.module("spatial", self)

        dataset = "ss" if s[0, 0].item() == 0 else "ref"
        theta = pyro.param("inverse_dispersion_" + dataset) #, 10.0 * x.new_ones(self.num_genes),
                           #constraint=constraints.positive)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            with poutine.scale(scale=anneal):
                z1 = pyro.sample("z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))

                if dataset == "ref":
                    y = pyro.sample("y", dist.OneHotCategorical(logits=x.new_zeros(self.num_classes)), obs=y)
                elif dataset == "ss" and r is not None:
                    logits = self.gp.model(r).transpose(-1, -2) if self.gp is not None else x.new_zeros(self.num_classes)
                    y = pyro.sample("y", dist.OneHotCategorical(logits=logits))
                elif dataset == "ss" and r is None:
                    y = pyro.sample("y", dist.OneHotCategorical(logits=x.new_zeros(self.num_classes)))

                z2_loc, z2_scale = self.z2_decoder(z1, y)
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

                l_scale = l_scale * x.new_ones(1)
                l = pyro.sample("l", dist.LogNormal(l_mean, l_scale).to_event(1))

                mu = self.x_decoder(z2, s)
                # TODO revisit this parameterization when https://github.com/pytorch/pytorch/issues/42449 is resolved
                nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
                x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits)
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, l_mean, l_scale, x, s, y=None, r=None, anneal=1.0):
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor * anneal):
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x, s)
            pyro.sample("l", dist.LogNormal(l_loc + l_mean, l_scale).to_event(1))
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            if r is not None:
                self.gp.guide(r)

            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                y = pyro.sample("y", y_dist)
            #else:
            #    classification_loss = y_dist.log_prob(y)
            #    pyro.factor("classification_loss", -self.alpha * classification_loss)

            z1_loc, z1_scale = self.z1_encoder(z2, y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

    def classifier_loss(self, x, s, y):
        z2_loc, z2_scale = self.z2l_encoder(x, s)[:2]
        z2 = torch.distributions.Normal(z2_loc, z2_scale).rsample()
        y_logits = self.classifier(z2)
        y_dist = dist.OneHotCategorical(logits=y_logits)
        return -y_dist.log_prob(y).mean()



def main(args):
    pyro.clear_param_store()
    pyro.util.set_rng_seed(args.seed)
    pyro.enable_validation(True)

    dataloader_train, dataloader_test, adata_ss, adata_ref = get_data(mock=False, batch_size=args.batch_size)

    num_genes = dataloader_train.X_ss.size(-1)

    #beta = float(args.batch_size) / float(dataloader.X_ss.size(0))
    spatial_gp = None # SpatialGP(num_classes=dataloader.num_classes, R_ss=dataloader.R_ss, beta=beta)
    spatial = Spatial(num_genes, dataloader_train.num_classes, spatial_gp,
                      scale_factor=1.0 / (args.batch_size * num_genes)).cuda()

    adam = torch.optim.Adam(list(spatial.parameters()) + list(pyro.get_param_store()._params.values()),
                            lr=args.learning_rate)
    #adam2 = torch.optim.Adam(spatial.classifier.parameters(),
    #                        lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.MultiStepLR(adam, [90], gamma=0.2)
    #optim = ClippedAdam({"lr": args.learning_rate, "clip_norm": 10.0})
    guide = config_enumerate(spatial.guide, "parallel", expand=True)
    #svi = SVI(spatial.model, guide, optim, TraceEnum_ELBO())
    loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss
    loss_fn2 = TraceEnum_ELBO(max_plate_nesting=2).loss

    ts = [time.time()]

    for epoch in range(args.num_epochs):
        losses = []

        for x, yr, l_mean, l_scale, dataset in dataloader_train.unlabeled_data():
            anneal = min(1.0, (epoch + 0.1) / 20.0)
            #if dataset == "ref":
            #    loss = loss_fn(spatial.model, guide, l_mean, l_scale, x, x.new_ones(x.size(0), 1), y=None, anneal=anneal)
            if dataset == "ss":
            #    if epoch < 500:
            #        yr = None
                loss = loss_fn(spatial.model, guide, l_mean, l_scale, x, x.new_zeros(x.size(0), 1), y=None, anneal=anneal)
            loss.backward()
            adam.step()
            adam.zero_grad()
            losses.append(loss.item())
            #spatial.eval()
            #spatial.train()

        ts.append(time.time())
        sched.step()

        if epoch % 2 == 0:
            spatial.eval()

            if 0:
                latent_rep = spatial.z2l_encoder(dataloader.X_ref, torch.ones(dataloader.num_ref_data, 1).cuda())[0]
                logits = spatial.classifier(latent_rep)
                y_hat = logits.max(-1)[1]
                probs = torch.softmax(logits, dim=-1).max(-1)[0]
                probs90 = (probs > 0.90).float().mean().item()
                probs80 = (probs > 0.80).float().mean().item()
                probs50 = (probs > 0.50).float().mean().item()
                print("Reference probs90 probs80 probs50: %.4f %.4f %.4f" % (probs90, probs80, probs50))
                accuracy = 100.0 * (y_hat == dataloader.Y_ref).sum().item() / float(y_hat.size(0))
                print("Reference accuracy: %.4f" % accuracy)

            if 0:
                latent_rep = spatial.z2l_encoder(dataloader.X_ss, torch.zeros(dataloader.num_ss_data, 1).cuda())[0]
                y_hat = spatial.classifier(latent_rep).max(-1)[1]
                print("SS label counts: ", np.bincount(y_hat.data.cpu().numpy()))

                theta_ref = pyro.param("inverse_dispersion_ref").data.cpu()
                theta_ss = pyro.param("inverse_dispersion_ss").data.cpu()
                print("theta_ref: %.3f %.3f %.3f   theta_ss: %.3f %.3f %.3f" % (
                      theta_ref.mean().item(), theta_ref.min().item(), theta_ref.max().item(),
                      theta_ss.mean().item(), theta_ss.min().item(), theta_ss.max().item()))

            #x = dataloader_test.X_ref
            #l_mean, l_scale = dataloader_train.l_mean_ref, dataloader_train.l_scale_ref
            #test_loss_ref = loss_fn2(spatial.model, guide, l_mean, l_scale, x, x.new_ones(x.size(0), 1), y=None)
            #test_loss_ref *= args.batch_size / 3000.0
            #theta_ref = pyro.param("inverse_dispersion_ref").data.cpu()

            x = dataloader_test.X_ss
            l_mean, l_scale = dataloader_train.l_mean_ss, dataloader_train.l_scale_ss
            test_loss_ss = loss_fn2(spatial.model, guide, l_mean, l_scale, x, x.new_zeros(x.size(0), 1), y=None)
            test_loss_ss *= args.batch_size / 1000.0
            theta_ss = pyro.param("inverse_dispersion_ss").data.cpu()

            #print("test_loss_ref: %.5f   theta_ref: %.3f %.3f %.3f     anneal: %.3f" % (test_loss_ref,
            #      theta_ref.mean().item(), theta_ref.min().item(), theta_ref.max().item(), anneal))
            print("test_loss_ss: %.5f   theta_ss: %.3f %.3f %.3f     anneal: %.3f" % (test_loss_ss,
                  theta_ss.mean().item(), theta_ss.min().item(), theta_ss.max().item(), anneal))

            if 0:
                log1 = spatial.gp.qf(dataloader.R_ss[0:2500])
                log2 = spatial.gp.qf(dataloader.R_ss[2500:5000])
                log3 = spatial.gp.qf(dataloader.R_ss[5000:7500])
                logits = torch.cat([log1, log2, log3])
                probs = torch.softmax(logits, dim=-1).mean(0)
                print("probs", probs.data.cpu().numpy())

            spatial.train()

        #closses = []
        #for it in range(5):
        #    for x, y, _, _, _ in dataloader.labeled_data():
        #        loss = spatial.classifier_loss(x, x.new_ones(x.size(0), 1), y)
        #        loss.backward()
        #        adam2.step()
        #        adam2.zero_grad()
        #        if it == 4:
        #            closses.append(loss.item())

        dt = 0.0 if epoch == 0 else ts[-1] - ts[-2]
        print("[Epoch %04d]  Loss: %.5f     [dt: %.3f]" % (epoch, np.mean(losses), dt))
        #print("[Epoch %04d]  Loss: %.5f   ClassLoss: %.5f    [dt: %.3f]" % (epoch, np.mean(losses), np.mean(closses), dt))

    # Done training
    spatial.eval()

    if 0:
        x = dataloader_train.X_ref
        y = dataloader_train.Y_ref.data.cpu().numpy()
        s = x.new_ones(x.size(0), 1)
        latent_rep = spatial.z2l_encoder(x, s)[0]
        adata_ref.obsm["X_scANVI"] = latent_rep.data.cpu().numpy()
        sc.pp.neighbors(adata_ref, use_rep="X_scANVI")
        sc.tl.umap(adata_ref)
        umap1, umap2 = adata_ref.obsm['X_umap'][:, 0], adata_ref.obsm['X_umap'][:, 1]

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.scatter(umap1, umap2, s=0.10, c=y, marker='.', alpha=0.8)
        ax.set_title('Learned Representation on Reference Data')
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')

        fig.tight_layout()
        plt.savefig('spatial_ref.pdf')

    x = dataloader_train.X_ss
    s = x.new_zeros(x.size(0), 1)
    latent_rep, _, l, _ = spatial.z2l_encoder(x, s)

    mu = spatial.x_decoder(latent_rep, s)
    theta = pyro.param("inverse_dispersion_ss")
    #nb_logits = (l * mu + spatial.epsilon).log() - (theta + spatial.epsilon).log()
    latent_rep = l * mu + spatial.epsilon
    print("latent_rep",latent_rep.shape)

    adata_ss.obsm["X_scANVI"] = latent_rep.data.cpu().numpy()
    sc.pp.neighbors(adata_ss, use_rep="X_scANVI")
    sc.tl.umap(adata_ss)
    umap1, umap2 = adata_ss.obsm['X_umap'][:, 0], adata_ss.obsm['X_umap'][:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(umap1, umap2, s=0.10, marker='.', alpha=0.8)
    ax.set_title('Learned Representation on SlideSeq Data')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

    fig.tight_layout()
    plt.savefig('spatial_ss.pdf')

    #latent_rep = spatial.z2l_encoder(dataloader.X_ss, torch.zeros(dataloader.num_ss_data, 1).cuda())[0]
    #y_logits = spatial.classifier(latent_rep)
    #y_probs = softmax(y_logits, dim=-1).data.cpu().numpy()
    #np.save("y_probs", y_probs)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-s', '--seed', default=0, type=int, help='rng seed')
    parser.add_argument('-n', '--num-epochs', default=40, type=int, help='number of training epochs')
    parser.add_argument('-bs', '--batch-size', default=400, type=int, help='mini-batch size')
    parser.add_argument('-lr', '--learning-rate', default=0.005, type=float, help='learning rate')
    args = parser.parse_args()

    main(args)
