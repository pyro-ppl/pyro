# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softplus, softmax, one_hot

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.importance import vectorized_importance_weights

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import scanpy as sc

from data import get_data


def make_fc(dims, dropout=0.0):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        #layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        #if dropout > 0.0:
        #    layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers[:-1])


def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


# p(x|z)
class XDecoder(nn.Module):
    def __init__(self, num_genes, z_dim, hidden_dims):
        super().__init__()
        dims = [z_dim] + hidden_dims + [num_genes]
        self.fc = make_fc(dims)

    def forward(self, z):
        mu = softmax(self.fc(z), dim=-1)
        return mu


# q(z, l | x, u)
class ZLEncoder(nn.Module):
    def __init__(self, num_genes, z_dim, hidden_dims, num_classes):
        super().__init__()
        dims = [num_classes + num_genes] + hidden_dims + [2 * z_dim]
        self.fc = make_fc(dims)

    def forward(self, x, u):
        x = torch.log(1.0 + x)
        x_u = torch.cat([x, u], dim=-1)
        h1, h2 = split_in_half(self.fc(x_u))
        z_loc, z_scale = h1, softplus(h2 / 2.0 - 1.0)
        return z_loc, z_scale


class Identifiable(nn.Module):
    def __init__(self, num_genes, num_classes, latent_dim=10, scale_factor=1.0,
                 l_loc=1.23, l_scale=1.23, num_cells=100, l_loc_init=None):
        self.num_genes = num_genes
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.scale_factor = scale_factor

        print("Initialized Identifiable with num_genes=%d, num_classes=%d, latent_dim=%d" % (num_genes,
               num_classes, latent_dim))
        super().__init__()

        self.decoder = XDecoder(num_genes=num_genes, hidden_dims=[100], z_dim=self.latent_dim)
        self.encoder = ZLEncoder(num_genes=num_genes, z_dim=self.latent_dim, hidden_dims=[100],
                                 num_classes=num_classes)

        self.epsilon = 1.0e-6

        pyro.param("inverse_dispersion", 10.0 * torch.ones(self.num_genes).cuda(),
                   constraint=constraints.positive)

        self.l_loc = l_loc
        self.l_scale = l_scale
        pyro.param("l_loc", l_loc_init.unsqueeze(-1).cuda())
        #pyro.param("l_loc", l_loc * torch.ones(num_cells, 1).cuda())
        pyro.param("l_scale", 0.1 * l_scale * torch.ones(num_cells, 1).cuda(),
                   constraint=constraints.positive)

        z_mean = pyro.param("z_mean", 0.1 * torch.randn(self.num_classes, self.latent_dim).cuda())
        z_scale = pyro.param("z_scale", torch.ones(self.num_classes, self.latent_dim).cuda(),
                             constraint=constraints.positive)

    def model(self, x, y, idx):
        theta = pyro.param("inverse_dispersion")
        z_mean = pyro.param("z_mean").index_select(-2, y)
        z_scale = pyro.param("z_scale").index_select(-2, y)

        with pyro.plate("batch", x.size(-2)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample("z", dist.Normal(z_mean, z_scale).to_event(1))

            l = pyro.sample("l", dist.LogNormal(self.l_loc, self.l_scale * x.new_ones(1)).to_event(1))

            mu = self.decoder(z)
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.NegativeBinomial(total_count=theta, logits=-nb_logits)
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, y, idx):
        l_loc = pyro.param("l_loc")[idx]
        l_scale = pyro.param("l_scale")[idx]
        z_prior_mean = pyro.param("z_mean").index_select(-2, y)
        with pyro.plate("batch", x.size(-2)), poutine.scale(scale=self.scale_factor):
            y = one_hot(y, num_classes=self.num_classes)
            z_loc, z_scale = self.encoder(x, y)
            l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            z = pyro.sample("z", dist.Normal(z_prior_mean + z_loc, z_scale).to_event(1))


def main(args):
    pyro.clear_param_store()
    pyro.util.set_rng_seed(args.seed)
    pyro.enable_validation(True)

    dataloader, adata_ss, adata_ref = get_data(mock=False, batch_size=args.batch_size)

    num_cells, num_genes = dataloader.X_ref.shape

    identifiable = Identifiable(num_genes, dataloader.num_classes,
                                scale_factor=1.0 / (args.batch_size * num_genes),
                                l_loc=dataloader.l_mean_ref, l_scale=dataloader.l_scale_ref,
                                l_loc_init=dataloader.X_ref.sum(-1).log().clone(),
                                num_cells=num_cells).cuda()

    adam = torch.optim.Adam(list(identifiable.parameters()) + list(pyro.get_param_store()._params.values()),
                            lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.MultiStepLR(adam, [50], gamma=0.2)
    #optim = ClippedAdam({"lr": args.learning_rate, "clip_norm": 10.0})
    #svi = SVI(spatial.model, identifiable.model.guide, optim, TraceEnum_ELBO())
    diff_loss_fn = Trace_ELBO(max_plate_nesting=1).differentiable_loss
    loss_fn = Trace_ELBO(max_plate_nesting=1).loss

    ts = [time.time()]

    for epoch in range(args.num_epochs):
        losses = []

        for x, y, idx in dataloader.labeled_data():
            loss = diff_loss_fn(identifiable.model, identifiable.guide, x, y, idx)
            loss.backward()
            adam.step()
            adam.zero_grad()
            losses.append(loss.item())

        #print(pyro.get_param_store()._params.keys())

        ts.append(time.time())
        sched.step()

        with torch.no_grad():
            if epoch % 2 == 0 and False:
                identifiable.eval()
                X = dataloader.X_ss[:10]
                X = X.unsqueeze(0).expand(19, X.size(0), X.size(1))
                y = torch.arange(19, device=x.device).unsqueeze(-1).expand(19, X.size(1))
                x = X.reshape(-1, X.size(-1))
                y = y.reshape(-1)
                l_mean, l_scale = dataloader.l_mean_ss, dataloader.l_scale_ss
                num_samples = 32
                model_trace = vectorized_importance_weights(identifiable.model, identifiable.guide, l_mean, x, y,
                                                            num_samples=num_samples, max_plate_nesting=1)[1]
                log_pz = model_trace.nodes['z']['unscaled_log_prob'].reshape((num_samples,) + X.shape[:2])
                log_px = model_trace.nodes['x']['unscaled_log_prob'].reshape((num_samples,) + X.shape[:2])
                log_pl = model_trace.nodes['l']['unscaled_log_prob'].reshape((num_samples,) + X.shape[:2])
                log_p = torch.logsumexp(log_pz + log_px + log_pl, dim=0)
                log_p, labels = log_p.max(0)
                labels = labels.data.cpu().numpy().tolist()
                labels = ["{:02}".format(l) for l in labels]
                print("NLL:  {:.5f}  labels: ".format(-log_p.mean().item() / num_genes), " ".join(labels))
                theta_ref = pyro.param("inverse_dispersion").data.cpu()

                #print("test_loss_ref: %.5f   theta_ref: %.3f %.3f %.3f     anneal: %.3f" % (test_loss_ref,
                #      theta_ref.mean().item(), theta_ref.min().item(), theta_ref.max().item(), anneal))
                print("theta_ref: %.3f %.3f %.3f" % (theta_ref.mean().item(), theta_ref.min().item(), theta_ref.max().item()))
                identifiable.train()

        dt = 0.0 if epoch == 0 else ts[-1] - ts[-2]
        print("[Epoch %04d]  Loss: %.5f     [dt: %.3f]" % (epoch, np.mean(losses), dt))
        if epoch % 3 == 0:
            print("l_loc: {:.5f}".format(pyro.param("l_loc").mean().item()),
                  " l_scale: {:.5f}".format(pyro.param("l_scale").mean().item()))

    # Done training
    identifiable.eval()

    x = dataloader.X_ref
    y = dataloader.Y_ref
    idx = torch.arange(x.size(0), device=x.device)
    latent_rep = identifiable.encoder(x, one_hot(y, num_classes=identifiable.num_classes))[0]
    adata_ref.obsm["X_scANVI"] = latent_rep.data.cpu().numpy()
    sc.pp.neighbors(adata_ref, use_rep="X_scANVI")
    sc.tl.umap(adata_ref)
    umap1, umap2 = adata_ref.obsm['X_umap'][:, 0], adata_ref.obsm['X_umap'][:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(umap1, umap2, s=0.10, c=y.data.cpu().numpy(), marker='.', alpha=0.8)
    ax.set_title('Learned Representation on Reference Data')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

    fig.tight_layout()
    plt.savefig('spatial_ref.pdf')

    import sys; sys.exit()

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
    parser.add_argument('-n', '--num-epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('-bs', '--batch-size', default=256, type=int, help='mini-batch size')
    parser.add_argument('-lr', '--learning-rate', default=0.005, type=float, help='learning rate')
    args = parser.parse_args()

    main(args)
