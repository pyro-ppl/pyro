# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
from pyro import distributions as dist


def model(data):
    a = pyro.sample("a", dist.Normal(0, 1))
    b = pyro.sample("b", dist.Normal(a, 1))
    c = pyro.sample("c", dist.Normal(a, b.exp()))
    d = pyro.sample("d", dist.Bernoulli(logits=c), obs=torch.tensor(0.0))

    with pyro.plate("p", len(data)):
        e = pyro.sample("e", dist.Normal(a, b.exp()))
        f = pyro.deterministic("f", e + 1)
        g = pyro.sample("g", dist.Delta(e + 1), obs=e + 1)
        h = pyro.sample("h", dist.Delta(e + 1))
        i = pyro.sample("i", dist.Normal(e, (f + g + h).exp()), obs=data)


obs = torch.ones(10)
g = pyro.render_model(
    model,
    model_args=(obs,),
    render_distributions=True,
    render_params=True,
    render_deterministic=True,
)
g.format = "png"
g.render("model", view=False)
