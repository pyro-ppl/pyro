from functools import partial
import pyro
import pyro.distributions as dist
import torch
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, TraceGraph_ELBO
from torch.distributions import constraints
import numpy as np


# both strategies work ok for small hidden_dim (e.g. 4) but
# only partial enumeration works well for large hidden_dim
hidden_dim = 64

def model(enumerate_x=False, enumerate_y=False):
    infer_x = {"enumerate": "parallel"} if enumerate_x else {}
    infer_y = {"enumerate": "parallel"} if enumerate_y else {}

    x_probs = 0.6 * torch.ones(hidden_dim)
    with pyro.plate("x_plate", hidden_dim):
        pyro.sample("x", dist.Bernoulli(probs=x_probs), infer=infer_x)

    y_probs = 0.3 * torch.ones(hidden_dim)
    with pyro.plate("y_plate", hidden_dim):
        pyro.sample("y", dist.Bernoulli(probs=y_probs), infer=infer_y)

    with pyro.plate("data", hidden_dim):
        pyro.sample("obs", dist.Normal(0.0, 1.0), obs=torch.ones(hidden_dim))

def y_guide():
    y_probs = pyro.param("y_probs", torch.rand(hidden_dim), constraint=constraints.unit_interval)
    with pyro.plate("y_plate", hidden_dim):
        pyro.sample("y", dist.Bernoulli(probs=y_probs))

def x_guide():
    x_probs = pyro.param("x_probs", torch.rand(hidden_dim), constraint=constraints.unit_interval)
    with pyro.plate("x_plate", hidden_dim):
        pyro.sample("x", dist.Bernoulli(probs=x_probs))

def xy_guide():
    x_guide()
    y_guide()


pyro.set_rng_seed(0)
optim = Adam({'lr': 0.005})
svi_x = SVI(partial(model, False, True), x_guide, optim, loss=TraceEnum_ELBO(max_plate_nesting=1, num_particles=1))
svi_y = SVI(partial(model, True, False), y_guide, optim, loss=TraceEnum_ELBO(max_plate_nesting=1, num_particles=1))

print("### Learn guide with partial enumeration strategy ###")

num_steps = 2000
losses = []
for step in range(num_steps):
    loss_x = svi_x.step()
    loss_y = svi_y.step()
    loss = 0.5 * (loss_x + loss_y)
    losses.append(loss)
    if step % 500 == 0 or step == num_steps - 1:
        running = 0.0 if step == 0 else np.mean(losses[-50:])
        mean_y, mean_x = pyro.param("y_probs").mean().item(), pyro.param("x_probs").mean().item()
        print("[step %d] %.3f %.3f" % (step, loss, running), "mean x/y probs: %.3f %.3f" % (mean_x, mean_y))

x_error = (0.6 - pyro.param("x_probs")).abs().mean().item()
y_error = (0.3 - pyro.param("y_probs")).abs().mean().item()
print("x/y error: %.4f %.4f" % (x_error, y_error))

pyro.clear_param_store()
pyro.set_rng_seed(0)
optim = Adam({'lr': 0.005})
svi = SVI(partial(model, False, False), xy_guide, optim, loss=TraceGraph_ELBO(max_plate_nesting=1, num_particles=1))

print("\n### Learn guide with fully stochastic strategy ###")

num_steps = 8000
losses = []
for step in range(num_steps):
    loss = svi.step()
    losses.append(loss)
    if step % 1000 == 0 or step == num_steps - 1:
        running = 0.0 if step == 0 else np.mean(losses[-50:])
        mean_y, mean_x = pyro.param("y_probs").mean().item(), pyro.param("x_probs").mean().item()
        print("[step %d] %.3f %.3f" % (step, loss, running), "mean x/y probs: %.3f %.3f" % (mean_x, mean_y))

x_error = (0.6 - pyro.param("x_probs")).abs().mean().item()
y_error = (0.3 - pyro.param("y_probs")).abs().mean().item()
print("x/y error: %.4f %.4f" % (x_error, y_error))
