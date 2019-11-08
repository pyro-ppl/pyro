import argparse

import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.contrib.timeseries import IndependentMaternGP

from gp_models import download_data


pyro.enable_validation(__debug__)


def main(args):
    download_data()
    data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
    print("[raw data shape] {}".format(data.shape))
    data = torch.tensor(data[::20, :-1]).double()
    data = data[:100, :1]  # choose a single output dimension
    print("[data shape after thinning] {}".format(data.shape))

    T, obs_dim = data.shape

    # standardize data
    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    torch.manual_seed(args.seed)

    gp = IndependentMaternGP(nu=1.5, obs_dim=obs_dim).double()

    def model():
        del gp.kernel.log_length_scale
        del gp.kernel.log_kernel_scale
        del gp.log_obs_noise_scale
        prior_dist = dist.Normal(0.0, 2 * torch.ones(obs_dim)).to_event(1)
        setattr(gp.kernel, "log_length_scale",
                pyro.sample("log_length_scale", prior_dist))
        setattr(gp.kernel, "log_kernel_scale",
                pyro.sample("log_kernel_scale", prior_dist))
        setattr(gp, "log_obs_noise_scale",
                pyro.sample("log_obs_noise_scale", prior_dist))

        factor = gp.log_prob(data).sum(-1)
        pyro.factor("y", factor)

    nuts_kernel = NUTS(model, max_tree_depth=4)
    mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=100, num_chains=1)
    mcmc.run()
    mcmc.summary(prob=0.5)


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.5.1')
    parser = argparse.ArgumentParser(description="contrib.timeseries hmc example usage")
    parser.add_argument("-s", "--seed", default=0, type=int)
    args = parser.parse_args()

    main(args)
