import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro import poutine
from pyro.contrib.autoguide import (ADVIDiagonalNormal, ADVIDiscreteParallel,  # noqa: F401
                                    ADVIMaster, ADVIMultivariateNormal)
import pyro.optim as optim
from utils import get_data


def model(K, alpha0, y):
    theta = pyro.sample("theta", dist.Dirichlet(alpha0 * torch.ones(K)))
    mu = pyro.sample("mu", dist.Normal(torch.zeros(K, y.shape[-1]),
                                       10. * torch.ones(K, y.shape[-1])).reshape(extra_event_dims=1))
    sigma = pyro.sample("sigma", dist.LogNormal(torch.ones(K, y.shape[-1]),
                                                torch.ones(K, y.shape[-1])).reshape(extra_event_dims=1))

    with pyro.iarange('data', len(y)):
        assign = pyro.sample('mixture', dist.Categorical(theta.unsqueeze(0).expand(len(y), K)))
        obs_dist = dist.Normal(mu[assign], sigma[assign]).reshape(extra_event_dims=1)
        pyro.sample('obs', obs_dist, obs=y[assign])


def main(args):
    advi = ADVIMaster(model)
    advi.add(ADVIDiagonalNormal(poutine.block(model, hide=["mixture"]))),
    advi.add(ADVIDiscreteParallel(poutine.block(model, expose=["mixture"])))

    adam = optim.Adam({'lr': 1e-3})
    svi = SVI(advi.model, advi.guide, adam, loss="ELBO", enum_discrete=True, max_iarange_nesting=1)
    for i in range(100):
        loss = svi.step(*args)
        print('loss=', loss)
        if i % 10 == 9:
            d = advi.parts[0].median()
            print({k: d[k].sum() / 200 for k in ["mu", "theta", "sigma"]})


if __name__ == "__main__":
    varnames = ["K", "alpha0", "y"]
    args = get_data("../../data/gmm_training_data.json", varnames)
    main(args)
