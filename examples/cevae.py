import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pyro
from pyro import poutine
from pyro.distributions import Bernoulli, Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam


def TODO(*args):
    raise NotImplementedError("TODO")


class FullyConnected(nn.Sequential):
    def __init__(sizes):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        super().__init__(*layers)


class DiagNormalNet(nn.Module):
    def __init__(self, sizes):
        self.fc = FullyConnected(sizes[:-1] + [sizes[-1] * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        d = loc_scale.size(-1) // 2
        loc = loc_scale[..., :d]
        scale = nn.functional.softplus(loc_scale[..., d:])
        return loc, scale


class BernoulliNet(nn.Module):
    def __init__(self, sizes):
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class Model(PyroModule):
    def __init__(self, args):
        self.latent_dim = args.latent_dim
        super().__init__()
        self.x_nn = DiagNormalNet([args.latent_dim] +
                                  [args.hidden_size] * args.num_layers +
                                  [args.feature_dim])
        self.y0_nn = BernoulliNet([args.latent_dim] +
                                  [args.hidden_size] * args.num_layers)
        self.y1_nn = BernoulliNet([args.latent_dim] +
                                  [args.hidden_size] * args.num_layers)
        self.t_nn = BernoulliNet(args.feature_dim)

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            z = pyro.sample("z", self.z_dist)
            x = pyro.sample("x", self.x_dist(z), obs=x)
            t = pyro.sample("t", self.t_dist(z), obs=t)
            y = pyro.sample("y", self.y_dist(t, z), obs=y)
        return y

    def z_dist(self):
        return Normal(0, 1).expand([self.latent_dim]).to_event(1)

    def x_dist(self, z):
        loc, scale = self.x_nn(z)
        return Normal(loc, scale).to_event(1)

    def y_dist(self, t, z):
        # Parameters are not shared among t values.
        logits0 = self.y0_nn(z)
        logits1 = self.y1_nn(z)
        logits = torch.where(t, logits1, logits0)
        return Bernoulli(logits=logits)

    def t_dist(self, x):
        logits = self.t_nn(x)
        return Bernoulli(logits=logits)


class Guide(PyroModule):
    def __init__(self):
        self.latent_dim = args.latent_dim
        super().__init__()
        self.t_nn = BernoulliNet(self.feature_dim)
        self.y_nn = FullyConnected([self.feature_dim] +
                                   [args.hidden_dim] * args.num_layers)
        self.y0_nn = BernoulliNet(args.hidden_dim)
        self.y1_nn = BernoulliNet(args.hidden_dim)
        self.z0_nn = DiagNormalNet([1 + args.feature_dim] +
                                   [args.hidden_dim] * args.num_layers +
                                   [args.latent_dim])
        self.z1_nn = DiagNormalNet([1 + args.feature_dim] +
                                   [args.hidden_dim] * args.num_layers +
                                   [args.latent_dim])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            t = pyro.sample("t", self.t_dist(x), obs=t)
            y = pyro.sample("y", self.y_dist(t, x), obs=y)
            pyro.sample("z", self.z_dist(t, y, x))

    def t_dist(self, x):
        logits = self.t_nn(x)
        return Bernoulli(logits=logits)

    def y_dist(self, t, x):
        # The first n-1 layers are identical for all t values.
        hidden = self.y_nn(x)
        # In the final layer params are not shared among t values.
        logits0 = self.y0_nn(hidden)
        logits1 = self.y1_nn(hidden)
        logits = torch.where(t, logits1, logits0)
        return Bernoulli(logits=logits)

    def z_dist(self, t, y, x):
        # Parameters are not shared among t values.
        y_x = torch.cat([y.unsqueeze(-1), x], dim=-1)
        loc0, scale0 = self.z0_nn(y_x)
        loc1, scale1 = self.z1_nn(y_x)
        loc = torch.where(t, loc1, loc0)
        scale = torch.where(t, scale1, scale0)
        return Normal(loc, scale)


class TraceCausalEffect_ELBO(Trace_ELBO):
    """
    The CEVAE objective (to maximize) is::

        -loss = ELBO + log q(t|x) + log q(y|t,x)
    """
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        # Construct -ELBO part.
        blocked_names = [name for name, site in guide_trace.nodes.items()
                         if site["type"] == "sample" and site["is_observed"]]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace)

        # Add log q terms.
        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob"]
            loss = loss - torch_item(log_q)
            surrogate_loss = surrogate_loss - log_q

        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))


def ite(model, guide, x, num_samples=100):
    """
    Computes Individual Treatment Effect for a batch of data ``x``.
    This has complexity ``O(len(x) * num_samples ** 2``.
    """
    with pyro.plate("guide_particles", num_samples, dim=-2):
        with poutine.block(hide=["y", "t"]):
            guide_trace = poutine.trace(guide).get_trace(x)
        with pyro.plate("model_particles", num_samples, dim=-3):
            with poutine.do(data=dict(t=0)):
                y0 = poutine.replay(model, guide_trace)(x).mean(0)
            with poutine.do(data=dict(t=2)):
                y1 = poutine.replay(model, guide_trace)(x).mean(0)
    return (y1 - y0).mean(0)


def train(args, x, t, y):
    model = Model(args)
    guide = Guide(args)

    dataset = TensorDataset(x, t, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_steps = args.num_epochs * len(dataloader)
    optim = ClippedAdam({"lr": args.learning_rate,
                         "weight_decay": args.weight_decay,
                         "lrd": args.learning_rate_decay ** (1 / num_steps)})
    svi = SVI(model, guide, optim, TraceCausalEffect_ELBO())
    for epoch in range(args.num_epochs):
        loss = 0
        for x, t, y in dataloader:
            loss += svi.step(x, t, y, size=len(dataset))
        print("epoch {: >3d} loss = {:0.6g}".format(loss / len(dataloader)))
    return model, guide


def generate_data(args):
    z = TODO()
    x = TODO(z)
    t = TODO(z)
    y = TODO(t, z)
    return x, t, y


def main(args):
    pyro.enable_validation(__debug__)

    # Generate synthetic data.
    pyro.set_rng_seed(args.seed)
    x, t, y = generate_data(args)

    # Train.
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    train(args, x, t, y)

    # Evaluate.
    TODO()


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.0.0')
    parser = argparse.ArgumentParser(description="Causal Effect Variational Autoencoder")
    parser.add_argument("--feature-dim", default=5, type=int)
    parser.add_argument("--latent-dim", default=20, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("--hidden-size", default=200, type=int)
    parser.add_argument("-n", "--num-epochs", default=1000, type=int)
    parser.add_argument("-b", "--batch-size", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=1234567890, type=int)
    args = parser.parse_args()
    main(args)
