from functools import partial
import pyro
import pyro.distributions as dist
import torch
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO
from torch.distributions import constraints
import numpy as np


hidden_dim = 1024

def model(enumerate_x=False, enumerate_y=False):
    infer_x = {"enumerate": "parallel"} if enumerate_x else {}
    infer_y = {"enumerate": "parallel"} if enumerate_y else {}

    x_probs = 0.6 * torch.ones(hidden_dim)
    with pyro.plate("x_plate", hidden_dim):
        x = pyro.sample("x", dist.Bernoulli(probs=x_probs), infer=infer_x)

    y_probs = 0.3 * torch.ones(hidden_dim)
    with pyro.plate("y_plate", hidden_dim):
        pyro.sample("y", dist.Bernoulli(probs=y_probs), infer=infer_y)

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

print("commencing learning with partial enumeration strategy")

num_steps = 2000
losses = []
for step in range(num_steps):
    loss_x = svi_x.step()
    loss_y = svi_y.step()
    loss = 0.5 * (loss_x + loss_y)
    losses.append(loss)
    if step % 200 == 0 or step == num_steps - 1:
        running = 0.0 if step == 0 else np.mean(losses[-50:])
        mean_y, mean_x = pyro.param("y_probs").mean().item(), pyro.param("x_probs").mean().item()
        print("[step %d] %.3f %.3f" % (step, loss, running), "mean x/y probs: %.3f %.3f" % (mean_x, mean_y))

print("y_probs[:3]: ", pyro.param("y_probs").data.cpu().numpy()[:3])
print("x_probs[:3]: ", pyro.param("x_probs").data.cpu().numpy()[:3])

pyro.clear_param_store()
pyro.set_rng_seed(0)
optim = Adam({'lr': 0.005})
svi = SVI(partial(model, False, False), xy_guide, optim, loss=TraceGraph_ELBO(max_plate_nesting=1, num_particles=1))

print("commencing learning with fully stochastic strategy")

num_steps = 8000
losses = []
for step in range(num_steps):
    loss = svi.step()
    losses.append(loss)
    if step % 400 == 0 or step == num_steps - 1:
        running = 0.0 if step == 0 else np.mean(losses[-50:])
        mean_y, mean_x = pyro.param("y_probs").mean().item(), pyro.param("x_probs").mean().item()
        print("[step %d] %.3f %.3f" % (step, loss, running), "mean x/y probs: %.3f %.3f" % (mean_x, mean_y))

print("y_probs[:3]: ", pyro.param("y_probs").data.cpu().numpy()[:3])
print("x_probs[:3]: ", pyro.param("x_probs").data.cpu().numpy()[:3])
