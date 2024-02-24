# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from functorch.dim import dims

import pyro
import pyro.distributions as dist
from pyro.contrib.named.infer.elbo import ELBO
from pyro.distributions.named import index_select

# i, j = dims(2)
# loc = torch.zeros(2, 3)[i, j]
# scale = torch.ones(2)[i]
# import pdb

# pdb.set_trace()
# normal = dist.Normal(loc, scale=torch.tensor(1.0), validate_args=False)
# k.size = 4
# normal.expand_named_shape([i, j, k])
# import pdb

# pdb.set_trace()
# normal.named_batch_shape
# x = normal.sample()
# log_prob_x = normal.log_prob(x)
# y = torch.randn(2)[i]
# log_prob_y = normal.log_prob(y)
# z = torch.randn(3, 4)[j, k]
# log_prob_z = normal.log_prob(z)
# dir = Dirichlet(torch.ones(3))

pyro.enable_validation(False)


# @config_enumerate
def model(data_dim, feature_dim, component_dim):
    data_plate = pyro.plate("data_plate", 6, dim=data_dim)
    feature_plate = pyro.plate("feature_plate", 5, dim=feature_dim)
    component_plate = pyro.plate("component_plate", 4, dim=component_dim)
    # component_plate = pyro.plate("component_plate", 4, dim=-1)
    with feature_plate, component_plate:
        p = pyro.sample("p", dist.Dirichlet(torch.ones(3)))
    with data_plate as idx:
        c = pyro.sample(
            "c", dist.Categorical(torch.ones(4).expand([data_dim.size, 4])[data_dim])
        )
        with feature_plate as vdx:  # Capture plate index.
            pc = index_select(p, dim=component_dim, index=c)
            # pc = p[c]
            x = pyro.sample(
                "x",
                dist.Categorical(pc),
                obs=torch.zeros(5, 6, dtype=torch.long)[vdx, idx],
            )
    print(f"    p.shape = {p.shape}")
    print(f"    c.shape = {c.shape}")
    print(f"  vdx.shape = {vdx.shape}")
    print(f"    pc.shape = {pc.shape}")
    print(f"    x.shape = {x.shape}")


def guide(data_dim, feature_dim, component_dim):
    data_plate = pyro.plate("data_plate", 6, dim=data_dim)
    feature_plate = pyro.plate("feature_plate", 5, dim=feature_dim)
    component_plate = pyro.plate("component_plate", 4, dim=component_dim)
    # component_plate = pyro.plate("component_plate", 4, dim=-1)
    with feature_plate, component_plate:
        pyro.sample(
            "p",
            dist.Dirichlet(
                torch.ones(3).expand([feature_dim.size, component_dim.size, 3])[
                    feature_dim, component_dim
                ]
            ),
        )
    with data_plate:
        pyro.sample(
            "c", dist.Categorical(torch.ones(4).expand([data_dim.size, 4])[data_dim])
        )


data_dim, feature_dim, component_dim = dims(3)
pyro.clear_param_store()
print("Sampling:")
print("Enumerated Inference:")
elbo = ELBO()
# model(data_dim, feature_dim, component_dim)
loss = elbo.loss(model, guide, data_dim, feature_dim, component_dim)
elbo_10 = ELBO(num_particles=10)
loss_10 = elbo_10.loss(model, guide, data_dim, feature_dim, component_dim)
elbo_100 = ELBO(num_particles=100)
loss_100 = elbo_100.loss(model, guide, data_dim, feature_dim, component_dim)
elbo_1000 = ELBO(num_particles=1000)
loss_1000 = elbo_1000.loss(model, guide, data_dim, feature_dim, component_dim)
