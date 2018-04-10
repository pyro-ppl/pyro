import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.util import log_sum_exp
from torch.distributions import constraints, transform_to
from pyro.infer import SVI, ADVIDiagonalNormal
import pyro.optim as optim
from pdb import set_trace as bb


def model(K, N, D, y, alpha0, alpha0_vec):
    theta = pyro.sample("theta", dist.Dirichlet(alpha0_vec))
    mu = pyro.sample("mu", dist.Normal(torch.zeros(K, D), torch.ones(K, D)*10.))
    sigma = pyro.sample("sigma", dist.LogNormal(torch.ones(K, D), torch.ones(K, D)))
    # sigma = transform_to(dist.Normal.arg_constraints['scale'])(sigma)

    ps = torch.zeros(K)

    for n in range(N):
        for k in range(K):
            ps[k] = torch.log(theta[k]) + torch.sum(dist.Normal(mu[k], sigma[k]).log_prob(y[n]))
        pyro.sample("ps[%d]" % (n), dist.Bernoulli(log_sum_exp(ps)), obs=(1))


def get_data(fname, varnames):
    import json
    with open(fname, "r") as f:
        j = json.load(f)
    d = {}
    for i in range(len(j[0])):
        var_name = j[0][i]
        if isinstance(j[1][i], int):
            val = j[1][i]
        else:
            val = torch.tensor(j[1][i])
        d[var_name] = val
    return tuple([d[k] for k in varnames]), d


def transformed_data(K, N, D, y, alpha0):
    alpha0_vec = torch.ones(K)*alpha0
    return alpha0_vec


def main(K, N, D, y, alpha0, alpha0_vec):
    advi = ADVIDiagonalNormal(model)
    adam = optim.Adam({'lr': 1e-3})
    svi = SVI(advi.model, advi.guide, adam, loss="ELBO")
    for i in range(100):
        loss = svi.step(K, N, D, y, alpha0, alpha0_vec)
        print('loss=', loss)
        if i % 5 == 0:
            d = advi.median()
            print({k: d[k] for k in ["mu", "theta", "sigma"]})


if __name__ == "__main__":
    varnames = ["K", "N", "D", "y", "alpha0"]
    (K, N, D, y, alpha0), data = get_data("data/training.data.json", varnames)
    alpha0_vec = transformed_data(K, N, D, y, alpha0)
    args = (K, N, D, y, alpha0, alpha0_vec)
    main(*args)
