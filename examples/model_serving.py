"""
This example demonstrates using torch.jit.ModuleScript for model deployment.
The Bayesian Regression model is taken from http://pyro.ai/examples/bayesian_regression.html.

In particular, the prediction function can be saved as a `torch.jit.ModuleScript`, and
be run separately as a C++ program without a Python runtime. To do so, we recommend the
following approach:

 - Limit state initialization in model / guide callable passed to `SVI`.
   e.g. initializing NNs. An easy way to ensure this is to inherit from
   `nn.Module`, and do this initialization in `__init__`.
 - Likewise, have a separate prediction function that derives from `nn.Module`.
   This lets us capture parameters from Pyro's global parameter store after training,
   and store as `nn.Parameter` attributes that can be serialized by `torch.jit.save`.
   Note that we use `torch.jit.trace_module` (instead of scripting), which supports
   many of the constructs needed for running doing predictions with Pyro models, like
   context managers and higher order functions.

**References:**

  - TorchScript FAQs, https://pytorch.org/docs/stable/jit.html#frequently-asked-questions
 """

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch

from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import Adam
from pyro import poutine
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal


DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


class Model(nn.Module):
    def __init__(self, p):
        super(Model, self).__init__()
        self.linear = nn.Linear(p, 1)
        weight_shape, bias_shape = self.linear.weight.shape, self.linear.bias.shape
        self.priors = {
            "weight": dist.Normal(torch.zeros(weight_shape), torch.ones(weight_shape)).to_event(2),
            "bias": dist.Normal(8. * torch.ones(bias_shape), 1000.)
        }

    def forward(self, x, y=None):
        lifted_module = pyro.random_module("reg_module", self.linear, self.priors)()
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        with pyro.plate("data", x.shape[0]):
            mean = lifted_module(x).squeeze(-1)
            return pyro.sample("obs", dist.Normal(mean, sigma), obs=y)


class Guide(nn.Module):
    def __init__(self, model, p):
        # Limit the scale parameter to stabilize learning.
        pyro.param("auto_scale", torch.ones(p + 2) * 1e-4,
                   constraint=dist.constraints.positive)
        self.autoguide = AutoDiagonalNormal(model)
        super(Guide, self).__init__()

    def forward(self, x, y=None):
        return self.autoguide(x, y)


class Predictor(nn.Module):
    def __init__(self, model, guide):
        super(Predictor, self).__init__()
        self.model = model
        self.guide = guide
        self.params = {}

    def save_params(self, x):
        """
        Capture parameters from the param store and store as `nn.Parameter` attributes.
        """
        with poutine.trace(param_only=True) as param_capture:
            self.guide(x)
        for name, site in param_capture.trace.nodes.items():
            self.params[name] = nn.Parameter(site["value"], requires_grad=False)

    def forward(self, x, y):
        # Replay using saved guide parameters
        guide_trace = poutine.trace(poutine.replay(self.guide, params=self.params)).get_trace(x, y=y)
        return poutine.replay(self.model, guide_trace)(x)


def print_mse(predict_fn, x_data, y_data, stage):
    err = 0.
    for _ in range(100):
        err += torch.mean((y_data - predict_fn(x_data, y_data)) ** 2)
    logging.info(f"\t MSE [{stage}]: {err / 100:.2f}")


def main(args):
    pyro.set_rng_seed(0)
    pyro.enable_validation()

    data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = data.loc[:, ["cont_africa", "rugged", "rgdppc_2000"]]
    df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
    df = df[np.isfinite(df.rgdppc_2000)]
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
    data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values,
                        dtype=torch.float)
    x_data, y_data = data[:, :-1], data[:, -1]
    pyro.clear_param_store()
    optim = Adam({"lr": 0.1})
    p = x_data.shape[-1]
    model = Model(p)
    guide = Guide(model, p)
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(model, guide, optim, loss=elbo)
    predict_fn = Predictor(model, guide)
    print_mse(predict_fn, x_data, y_data, "before training")

    # Run inference
    for j in range(args.num_iter):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            logging.info("[iter %04d] loss: %.4f" % (j, loss / len(data)))

    print_mse(predict_fn, x_data, y_data, "after training")

    # Store parameters from the param store for jit.save
    predict_fn.save_params(x_data)
    predict_module = torch.jit.trace_module(predict_fn, {"forward": (x_data, y_data)}, check_trace=False)
    torch.jit.save(predict_module, os.path.join(DATA_DIR, 'reg_predict.pt'))

    # Test model load
    pyro.clear_param_store()
    predict_fn = torch.jit.load(os.path.join(DATA_DIR, 'reg_predict.pt'))
    print_mse(predict_fn, x_data, y_data, "after model load")


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.4.1')
    parser = argparse.ArgumentParser(description="Model serving demo.")
    parser.add_argument("-n", "--num-iter", default=6000, type=int)
    parser.add_argument('--jit', action='store_true')
    main(parser.parse_args())
