"""
This example implements the Causal Effect Variational Autoencoder [1].

This demonstrates a number of innovations including:
- a generative model for causal effect inference with hidden confounders;
- a model and guide with twin neural nets to allow imbalanced treatment; and
- a custom training loss that includes both ELBO terms and extra terms needed
  to train the guide to be able to answer counterfactual queries.

**References**

[1] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling (2017).
    Causal Effect Inference with Deep Latent-Variable Models.
    http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf
    https://github.com/AMLab-Amsterdam/CEVAE
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam


class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        super().__init__(*layers)


class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [sizes[-1] * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        d = loc_scale.size(-1) // 2
        loc = loc_scale[..., :d]
        scale = nn.functional.softplus(loc_scale[..., d:]).clamp(min=1e-10)
        return loc, scale


class BernoulliNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        t = dist.Bernoulli(logits=net(z)).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class Model(PyroModule):
    """
    Generative model for a causal model with latent confounder ``z`` and binary
    treatment ``t``::

        z ~ p(z)      # latent confounder
        x ~ p(x|z)    # partial noisy observation of z
        t ~ p(t|z)    # treatment, whose application is biased by z
        y ~ p(y|t,z)  # outcome

    Each of these distributions is defined by a neural network.  The ``y``
    distribution is defined by a disjoint pair of neural networks defining
    ``p(y|t=0,z)`` and ``p(y|t=1,z)``; this allows highly imbalanced treatment.
    """
    def __init__(self, args):
        self.latent_dim = args.latent_dim
        super().__init__()
        self.x_nn = DiagNormalNet([args.latent_dim] +
                                  [args.hidden_dim] * args.num_layers +
                                  [args.feature_dim])
        self.y0_nn = BernoulliNet([args.latent_dim] +
                                  [args.hidden_dim] * args.num_layers)
        self.y1_nn = BernoulliNet([args.latent_dim] +
                                  [args.hidden_dim] * args.num_layers)
        self.t_nn = BernoulliNet([args.latent_dim])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            z = pyro.sample("z", self.z_dist())
            x = pyro.sample("x", self.x_dist(z), obs=x)
            t = pyro.sample("t", self.t_dist(z), obs=t)
            y = pyro.sample("y", self.y_dist(t, z), obs=y)
        return y

    def z_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)

    def x_dist(self, z):
        loc, scale = self.x_nn(z)
        return dist.Normal(loc, scale).to_event(1)

    def y_dist(self, t, z):
        # Parameters are not shared among t values.
        logits0 = self.y0_nn(z)
        logits1 = self.y1_nn(z)
        logits = torch.where(t.bool(), logits1, logits0)
        return dist.Bernoulli(logits=logits)

    def t_dist(self, z):
        logits = self.t_nn(z)
        return dist.Bernoulli(logits=logits)


class Guide(PyroModule):
    """
    Inference model for causal effect estimation with latent confounder ``z``
    and binary treatment ``t``::

        t ~ p(t|x)      # treatment
        y ~ p(y|t,x)    # outcome
        z ~ p(t|y,t,x)  # latent confounder, an embedding

    Each of these distributions is defined by a neural network.  The ``y`` and
    ``z`` distributions are defined by disjoint pairs of neural networks
    defining ``p(-|t=0,...)`` and ``p(-|t=1,...)``; this allows highly
    imbalanced treatment.
    """
    def __init__(self, args):
        self.latent_dim = args.latent_dim
        super().__init__()
        self.t_nn = BernoulliNet([args.feature_dim])
        self.y_nn = FullyConnected([args.feature_dim] +
                                   [args.hidden_dim] * args.num_layers)
        self.y0_nn = BernoulliNet([args.hidden_dim])
        self.y1_nn = BernoulliNet([args.hidden_dim])
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
            pyro.sample("z", self.z_dist(y, t, x))

    def t_dist(self, x):
        logits = self.t_nn(x)
        return dist.Bernoulli(logits=logits)

    def y_dist(self, t, x):
        # The first n-1 layers are identical for all t values.
        hidden = self.y_nn(x)
        # In the final layer params are not shared among t values.
        logits0 = self.y0_nn(hidden)
        logits1 = self.y1_nn(hidden)
        logits = torch.where(t.bool(), logits1, logits0)
        return dist.Bernoulli(logits=logits)

    def z_dist(self, y, t, x):
        # Parameters are not shared among t values.
        y_x = torch.cat([y.unsqueeze(-1), x], dim=-1)
        loc0, scale0 = self.z0_nn(y_x)
        loc1, scale1 = self.z1_nn(y_x)
        loc = torch.where(t.bool().unsqueeze(-1), loc1, loc0)
        scale = torch.where(t.bool().unsqueeze(-1), scale1, scale0)
        return dist.Normal(loc, scale).to_event(1)


class TraceCausalEffect_ELBO(Trace_ELBO):
    """
    The CEVAE objective (to maximize) is [1]::

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
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            loss = loss - torch_item(log_q)
            surrogate_loss = surrogate_loss - log_q

        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))


def ite(model, guide, x, num_samples=100):
    r"""
    Computes Individual Treatment Effect for a batch of data ``x``.

    .. math::

        ITE(x) = \mathbb E[ \mathbf y \mid \mathbf X=x, do(\mathbf t=1)]
               - \mathbb E[ \mathbf y \mid \mathbf X=x, do(\mathbf t=0)]

    This has complexity ``O(len(x) * num_samples ** 2``.

    :param Model model: A trained CEVAE model.
    :param Guide guide: A trained CEVAE guide.
    :param torch.Tensor x: A batch of data.
    :param int num_samples: The number of monte carlo samples.
    :return: A ``len(x)``-sized tensor of estimated effects.
    :rtype: torch.Tensor
    """
    with pyro.plate("num_particles", num_samples, dim=-2):
        with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
            guide(x)
        with poutine.do(data=dict(t=torch.tensor(0.))):
            y0 = poutine.replay(model, tr.trace)(x)
        with poutine.do(data=dict(t=torch.tensor(1.))):
            y1 = poutine.replay(model, tr.trace)(x)
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
            loss += svi.step(x, t, y, size=len(dataset)) / len(dataset)
        print("epoch {: >3d} loss = {:0.6g}".format(epoch, loss / len(dataloader)))
    return model, guide


def generate_data(args):
    """
    This implements the generative process of [1], but using larger feature and
    latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    """
    z = dist.Bernoulli(0.5).sample([args.num_data])
    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([args.feature_dim]).t()
    t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
    y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()

    # Compute true ite for evaluation (via Monte Carlo approximation).
    t0_t1 = torch.tensor([[0.], [1.]])
    y_t0, y_t1 = dist.Bernoulli(logits=3 * (z + 2 * (2 * t0_t1 - 2))).mean
    true_ite = y_t1 - y_t0
    return x, t, y, true_ite


def main(args):
    pyro.enable_validation(__debug__)

    # Generate synthetic data.
    pyro.set_rng_seed(args.seed)
    x_train, t_train, y_train, _ = generate_data(args)

    # Train.
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    model, guide = train(args, x_train, t_train, y_train)

    # Evaluate.
    x_test, t_test, y_test, true_ite = generate_data(args)
    true_ate = true_ite.mean()
    print("true ATE = {:0.3g}".format(true_ate.item()))
    naive_ate = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print("naive ATE = {:0.3g}".format(naive_ate))
    est_ite = ite(model, guide, x_test, num_samples=10)
    est_ate = est_ite.mean()
    print("estimated ATE = {:0.3g}".format(est_ate.item()))


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.0.0')
    parser = argparse.ArgumentParser(description="Causal Effect Variational Autoencoder")
    parser.add_argument("--num-data", default=1000, type=int)
    parser.add_argument("--feature-dim", default=5, type=int)
    parser.add_argument("--latent-dim", default=20, type=int)
    parser.add_argument("--hidden-dim", default=200, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("-n", "--num-epochs", default=10, type=int)
    parser.add_argument("-b", "--batch-size", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=1234567890, type=int)
    args = parser.parse_args()
    main(args)
