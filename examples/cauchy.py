import argparse
import logging

import torch

import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter, TraceTMC_ELBO, config_enumerate

from pyro.ops.ssm_gp import MaternKernel
from pyro.ops.tensor_utils import block_diag_embed
from pyro.contrib.timeseries import IndependentMaternGP
import torch.distributions.constraints as constraints


logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)



class SimpleTimeSeriesModel:

    def __init__(self, nu=1.5):
        self.kernel = MaternKernel(nu=nu, num_gps=1, length_scale_init=torch.tensor([0.3]))
        self.sigma_obs = torch.tensor(0.1)
        self.dt = 1.0
        self.trans_matrix, self.process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        self.trans_matrix, self.process_covar = self.trans_matrix[0], self.process_covar[0]

    def init(self):
        self.t = 0
        init_dist = dist.MultivariateNormal(torch.zeros(self.kernel.state_dim),
                                            self.kernel.stationary_covariance().squeeze(-3))
        return pyro.sample("z_0", init_dist)

    def step(self, z_prev, y=None):
        self.t += 1
        z = pyro.sample("z_{}".format(self.t),
                        dist.MultivariateNormal(z_prev.matmul(self.trans_matrix),
                                                self.process_covar))
        y = pyro.sample("y_{}".format(self.t),
                        dist.Normal(z[..., 0], self.sigma_obs),
                        obs=y)
        return z


class SimpleTimeSeriesGuide:

    def __init__(self, model):
        self.model = model

    def init(self):
        self.t = 0
        scale = pyro.param("scale_{}".format(self.t), 2.0 * torch.ones(self.model.kernel.state_dim), constraint=constraints.positive)
        return pyro.sample("z_0", dist.Normal(torch.zeros(self.model.kernel.state_dim), scale).to_event(1))

    def step(self, z_prev, y=None):
        self.t += 1
        return pyro.sample("z_{}".format(self.t),
                           dist.Normal(z_prev.matmul(self.model.trans_matrix), 2.0).to_event(1))


def smc_run(args, ys):
    assert False
    model = SimpleHarmonicModel()
    guide = SimpleHarmonicModel_Guide(model)

    smc = SMCFilter(model, guide, num_particles=args.num_particles, max_plate_nesting=0)

    logging.info("Filtering")

    smc.init()
    for y in ys:
        smc.step(y)

    logging.info("At final time step:")
    z = smc.get_empirical()["z"]
    #print("z\n",z._samples[:, 0].data.numpy().tolist())
    logging.info("ys: {}".format(ys[-5:]))
    logging.info("mean: {}".format(z.mean))
    logging.info("std: {}".format(z.variance ** 0.5))


def tmc_run(args, ys):
    model = SimpleTimeSeriesModel()
    guide = SimpleTimeSeriesGuide(model)

    tmc = TraceTMC_ELBO(max_plate_nesting=0)

    def tmc_model(ys):
        z = model.init()
        for y in pyro.markov(ys):
            z = model.step(z, y)

    @config_enumerate(default="parallel", num_samples=args.num_particles, expand=False)
    def tmc_guide(ys):
        z = guide.init()
        for y in pyro.markov(ys):
            z = guide.step(z, y)


    optim = pyro.optim.Adam({'lr': 0.005})
    svi = pyro.infer.SVI(tmc_model, tmc_guide, optim, tmc)

    for step in range(10):
        logp = svi.step(ys)
        logging.info("[Step {}]  loss: {:.4f}".format(step, logp))

    gp = IndependentMaternGP(nu=1.5, obs_dim=1,
                             length_scale_init=model.kernel.length_scale).double()
    exact_log_prob = gp.log_prob(ys.unsqueeze(-1).double())
    logging.info("exact log prob: {:.4f}".format(exact_log_prob.item()))



def main(args):
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    logging.info("Generating data")
    T = args.num_timesteps
    ts = 3.0 * torch.arange(T).float() / T
    ys = torch.sin(2.0 * ts)

    if args.inference == "smc":
        smc_run(args, ys)
    elif args.inference == "tmc":
        tmc_run(args, ys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Harmonic Oscillator w/ SMC Filtering Inference")
    parser.add_argument("-n", "--num-timesteps", default=4, type=int)
    parser.add_argument("-p", "--num-particles", default=50, type=int)
    parser.add_argument("--process-noise", default=1., type=float)
    parser.add_argument("--measurement-noise", default=1., type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--inference", default="smc", type=str)
    args = parser.parse_args()
    main(args)
