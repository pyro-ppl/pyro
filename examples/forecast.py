import logging
import math

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.examples.bart import load_bart_od, load_fake_od
from pyro.distributions.stable import StableReparameterizer
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn.module import PyroModule, PyroParam
from pyro.optim import ClippedAdam


def linear_ode(ks, xs):
    """
    Integrate the linear difference equation::

        ys[0] = 0
        ys[t] = ys[t-1] * ks[t] + xs[t]
    """
    ys = [torch.zeros_like(xs[0])]
    for k, x in zip(ks, xs):
        ys.append(ys[-1] * k + x)
    return torch.stack(ys)


class Model(PyroModule):
    def __init__(self, dt):
        super().__init__()
        self.dt = torch.as_tensor(dt)
        assert not self.dt.shape
        self.stability = PyroParam(torch.tensor(1.9),
                                   constraint=constraints.positive)

    def forward(self, time, features, observations):
        assert len(time) == len(features)
        assert len(time) == len(observations)
        time_plate = pyro.plate("time", len(time), dim=-2)
        stability = self.stability
        scale = self.dt.pow(stability.reciprocal())
        log_decay = pyro.sample("log_decay",
                                dist.LogNormal(math.log(1 / local_timescale), 1))
        decay = log_decay.neg().exp()

        with pyro.plate("latent_plate", self.latent_dim):
            global_skew = pyro.sample("global_skew", dist.Uniform(-1, 1))
            global_scale = scale * pyro.sample("global_scale", dist.LogNormal(0, 1))
            with time_plate:
                ddt_growth = pyro.sample("ddt_growth", dist.Normal(0, dt.sqrt()))
                growth = ddt_growth.cumsum()
                ddt_global_trend = pyro.sample("ddt_global_trend",
                                               dist.Stable(stability, global_skew, global_scale))
                global_trend = linear_ode(growth, ddt_global_trend)

            local_skew = pyro.sample("local_skew", dist.Uniform(-1, 1))
            local_scale = scale * pyro.sample("local_scale", dist.LogNormal(0, 1))
            with time_plate:
                ddy_local_trend = pyro.sample("ddy_local_trend",
                                              dist.Stable(stability, local_skew, local_scale))
                local_trend = linear_ode(decay.expand(ddy_local_trend.shape[:-1]),
                                         ddy_local_trend_shape)

        latent_trend = global_trend + local_trend
        obs_trend = self.linear_map(latent_trend)

        with pyro.plate("obs_plate", self.obs_dim):
            noise_skew = pyro.sample("noise_skew", dist.Uniform(-1, 1))
            noise_scale = pyro.sample("noise_scale", dist.LogNormal(0, 1))
            with time_plate:
                noise = pyro.sample("noise",
                                    dist.Stable(stability, noise_skew, noise_scale))
                # TODO add seasonal component
                rate = (obs_trend + noise).softmax().add(1e-6)
                return pyro.sample("obs", dist.Poisson(rate), obs=observations)


def config_reparam(site):
    if isinstance(site["fn"], dist.Stable):
        return {"reparam": StableReparameterizer()}
    return {}


def fit(args, model, guide, data):
    optim = ClippedAdam({"lr": args.learning_rate,
                         "lrd": args.learning_rate_decay ** (1 / args.num_steps)})
    elbo = Trace_ELBO(max_plate_nesting=2)
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for step in range(args.num_steps):
        loss = svi.step(time, features, observations)
        losses.append(loss)
        logging.info("step {} loss = {}".format(step, loss))
    return losses


def main(args):
    data = load_fake_od() if args.test else load_bart_od()
    model = Model(args)
    model = poutine.reparam(poutine.infer_config(model, config_reparam))
    guide = AutoDiagonalNormal(model)
    fit(args, model, guide, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multivariate forecasting")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)
