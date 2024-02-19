import torch
import torch.distributions.constraints as constraints
from functorch.dim import Dim, dims

import pyro
from pyro import poutine
from pyro.contrib.named.infer.elbo import ELBO
from pyro.distributions.named import (
    Categorical,
    Dirichlet,
    LogNormal,
    Normal,
    index_select,
)
from pyro.infer import config_enumerate

# from pyro.ops.indexing import Vindex

# make_dist(dist.Normal)
i = Dim("i")
j = Dim("j")
k = Dim("k")
# i, j = dims(2)
loc = torch.zeros(2, 3)[i, j]
scale = torch.ones(2)[i]
normal = Normal(loc, scale=scale, validate_args=True)
x = normal.sample()
log_prob_x = normal.log_prob(x)
y = torch.randn(2)[i]
log_prob_y = normal.log_prob(y)
z = torch.randn(3, 4)[j, k]
log_prob_z = normal.log_prob(z)
dir = Dirichlet(torch.ones(3))


@config_enumerate
def model(i, j, k):
    data_plate = pyro.plate("data_plate", 6, dim=i)
    feature_plate = pyro.plate("feature_plate", 5, dim=j)
    component_plate = pyro.plate("component_plate", 4, dim=k)
    with feature_plate:
        with component_plate:
            p = pyro.sample("p", Dirichlet(torch.ones(3)))
    with data_plate as idx:
        c = pyro.sample("c", Categorical(torch.ones(4)))
        with feature_plate as vdx:  # Capture plate index.
            pc = index_select(p, dim=k, index=c)
            x = pyro.sample(
                "x",
                Categorical(pc),
                obs=torch.zeros(5, 6, dtype=torch.long)[vdx, idx],
            )
    print(f"    p.shape = {p.shape}")
    print(f"    c.shape = {c.shape}")
    print(f"  vdx.shape = {vdx.shape}")
    print(f"    pc.shape = {pc.shape}")
    print(f"    x.shape = {x.shape}")


def guide(i, j, k):
    feature_plate = pyro.plate("feature_plate", 5, dim=j)
    component_plate = pyro.plate("component_plate", 4, dim=k)
    with feature_plate, component_plate:
        pyro.sample("p", Dirichlet(torch.ones(3)))


pyro.clear_param_store()
print("Sampling:")
# model()
print("Enumerated Inference:")
data, feature, component = dims(3)
elbo = ELBO()
loss = elbo.loss(model, guide, data, feature, component)
elbo_10 = ELBO(num_particles=10)
loss_10 = elbo_10.loss(model, guide, data, feature, component)
elbo_100 = ELBO(num_particles=100)
loss_100 = elbo_100.loss(model, guide, data, feature, component)
elbo_1000 = ELBO(num_particles=1000)
loss_1000 = elbo_1000.loss(model, guide, data, feature, component)
import pdb

pdb.set_trace()
pass

# poutine.enum(model)()

# Examples

def model(data):
    ...
        mu = pyro.sample(
        "mu",
        MultivariateNormal(
            torch.zeros(2, device=device), 5 * torch.eye(2, device=device)
        )
        .expand([T])
        .to_event(1),
     )
     ...
         mu_vindex = Vindex(mu)[..., z, :]. # no if/else

def model(data):
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T):
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2, device=device), 5 * torch.eye(2, device=device)))

    with pyro.plate("data", N):
        #dim=-4 is an enumeration dim for z (e.g. T = 6 clusters)
        #dim=-3 is a particle vectorization (e.g. num_particles = 10 particles)
        #dim=-2 is allocated for "data" plate (1 value broadcasted over a batch)
        #dim=-1 is allocated as event dimension (2 values)
        z = pyro.sample("z", Categorical(mix_weights(beta).unsqueeze(-2)))
        pyro.sample("obs", MultivariateNormal(index_select(mu, "mu_plate", z),
                                              torch.eye(2, device=device)), obs=data)

########################################        
def guide(data):
    kappa = pyro.param('kappa', lambda: Uniform(torch.tensor(0., device=device), torch.tensor(2., device=device)).sample([T-1]),
                                        constraint=constraints.positive)
    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2, device=device), 3 * torch.eye(2, device=device)).sample([T]))
    phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T, device=device)).sample([N]), constraint=constraints.simplex)

    with pyro.plate("beta_plate", T-1, device=device):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1, device=device), kappa))

    with pyro.plate("mu_plate", T, device=device):
        q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2, device=device)))

    with pyro.plate("data", N, device=device):
        z = pyro.sample("z", Categorical(phi))

def guide(data):
    # no plate here, use to_event instead
    q_mu = pyro.sample(
        "mu",
        MultivariateNormal(tau, torch.eye(2, device=device)).to_event(1),
    )