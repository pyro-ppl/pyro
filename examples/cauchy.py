import argparse
import logging
import numpy as np

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import TraceTMC_ELBO, config_enumerate
from pyro.ops.ssm_gp import MaternKernel
from pyro.contrib.timeseries import IndependentMaternGP
from pyro.nn.module import clear


logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


class SimpleTimeSeriesModel:
    def __init__(self, nu=1.5):
        self.kernel = MaternKernel(nu=nu, num_gps=1, length_scale_init=torch.tensor([1.5]),
                                   kernel_scale_init=torch.tensor([0.5]))
        self.dt = 1.0

    def init(self):
        self.t = 0
        self.trans_matrix, self.process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        self.trans_matrix, self.process_covar = self.trans_matrix[0], self.process_covar[0]
        init_dist = dist.MultivariateNormal(torch.zeros(self.kernel.state_dim),
                                            self.kernel.stationary_covariance().squeeze(-3))
        return pyro.sample("z_0", init_dist)

    def step(self, z_prev, y=None):
        self.t += 1
        sigma_obs = pyro.param("sigma_obs", torch.tensor(0.1), constraint=constraints.positive)
        z = pyro.sample("z_{}".format(self.t),
                        dist.MultivariateNormal(z_prev.matmul(self.trans_matrix),
                                                self.process_covar))
        pyro.sample("y_{}".format(self.t), dist.Normal(z[..., 0], sigma_obs),
                    obs=y)
        return z


class SimpleTimeSeriesGuide:
    def __init__(self, model):
        self.model = model

    def init(self):
        self.t = 0
        scale = pyro.param("scale_{}".format(self.t), 0.5 * torch.ones(self.model.kernel.state_dim),
                           constraint=constraints.positive)
        loc = pyro.param("loc_{}".format(self.t), torch.zeros(self.model.kernel.state_dim))
        return pyro.sample("z_0", dist.Normal(loc, scale).to_event(1))

    def step(self, z_prev, y=None):
        self.t += 1
        loc = pyro.param("loc_{}".format(self.t), torch.zeros((self.model.kernel.state_dim)))
        scale = pyro.param("scale_{}".format(self.t), 0.5 * torch.ones(self.model.kernel.state_dim),
                           constraint=constraints.positive)
        return pyro.sample("z_{}".format(self.t), dist.Normal(loc, scale).to_event(1))


def tmc_run(args, ys):
    model = SimpleTimeSeriesModel()
    guide = SimpleTimeSeriesGuide(model)

    tmc = TraceTMC_ELBO(max_plate_nesting=0)

    def tmc_model(ys):
        z = model.init()
        for y in pyro.markov(ys):
            z = model.step(z, y)

    @config_enumerate(default="parallel", num_samples=args.num_particles, expand=False, tmc=args.tmc_strategy)
    def tmc_guide(ys):
        z = guide.init()
        for y in pyro.markov(ys):
            z = guide.step(z, y)

    def exact_eval(ys):
        gp = IndependentMaternGP(nu=1.5, obs_dim=1,
                                 length_scale_init=model.kernel.length_scale,
                                 kernel_scale_init=model.kernel.kernel_scale,
                                 obs_noise_scale_init=pyro.param("sigma_obs").detach().unsqueeze(-1)).double()
        log_prob = gp.log_prob(ys.unsqueeze(-1).double()).item()
        clear(gp)
        return log_prob

    svi_model, svi_guide = None, None
    if args.train == "iwae":
        model_optim = pyro.optim.ClippedAdam({'lr': 0.001, 'betas': (0.90, 0.999), 'clip_norm': 1.0})
        svi_model = pyro.infer.SVI(tmc_model, tmc_guide, model_optim, tmc.differentiable_loss)
    elif args.train == "rws":
        model_optim = pyro.optim.ClippedAdam({'lr': 0.001, 'betas': (0.90, 0.999), 'clip_norm': 1.0})
        svi_model = pyro.infer.SVI(tmc_model, poutine.block(tmc_guide, hide_types=["param"]), model_optim,
                                   tmc.differentiable_loss)
        guide_optim = pyro.optim.ClippedAdam({'lr': 0.001, 'betas': (0.90, 0.999), 'clip_norm': 1.0})
        svi_guide = pyro.infer.SVI(poutine.block(tmc_model, hide_types=["param"]), tmc_guide, guide_optim,
                                   tmc.wake_phi_loss)
    elif args.train == "model":  # no guide updates
        model_optim = pyro.optim.ClippedAdam({'lr': 0.001, 'betas': (0.90, 0.999), 'clip_norm': 1.0})
        svi_model = pyro.infer.SVI(tmc_model, poutine.block(tmc_guide, hide_types=["param"]), model_optim,
                                   tmc.differentiable_loss)
    else:
        raise ValueError

    pyro.param("sigma_obs", torch.tensor(0.1), constraint=constraints.positive)
    logging.info("Initial exact log prob: {:.6f}".format(exact_eval(ys)))

    num_steps = 300

    for step in range(num_steps):
        logp = svi_model.step(ys)
        if svi_guide is not None:
            svi_guide.step(ys)
        if step % 10 == 0 or step == num_steps - 1:
            frmt = "[Step {}]  loss: {:.4f}  exact: {:.4f}  lengthscale: {:.3f}  kernelscale: {:.3f}  sigmaobs: {:.3f}"
            logging.info(frmt.format(step, logp, exact_eval(ys), model.kernel.length_scale.item(),
                                     model.kernel.kernel_scale.item(), pyro.param("sigma_obs").item()))

    tmc_estimates = [-tmc.loss(tmc_model, tmc_guide, ys) for _ in range(100)]
    logging.info("TMC-estimated log prob: {:.4f} +- {:.4f}".format(np.mean(tmc_estimates), np.std(tmc_estimates)))
    logging.info("Final exact log prob: {:.6f}".format(exact_eval(ys)))


def exact_run(args, ys):

    gp_model = IndependentMaternGP(nu=1.5, obs_dim=1,
                                   length_scale_init=torch.tensor([1.5]),
                                   kernel_scale_init=torch.tensor([0.5]),
                                   obs_noise_scale_init=torch.tensor([0.1]))
    model_optim = pyro.optim.clipped_adam.ClippedAdam(
        gp_model.parameters(), **{'lr': 0.001, 'betas': (0.90, 0.999), 'clip_norm': 1.0})

    logging.info("Initial exact log prob: {:.6f}".format(
        gp_model.log_prob(ys.unsqueeze(-1)).item()))

    num_steps = 300

    for step in range(num_steps):
        log_prob = -gp_model.log_prob(ys.unsqueeze(-1))
        log_prob.backward()
        model_optim.step()
        logp = log_prob.item()
        if step % 10 == 0 or step == num_steps - 1:
            frmt = "[Step {}]  loss: {:.4f}  exact: {:.4f}  lengthscale: {:.3f}  kernelscale: {:.3f}  sigmaobs: {:.3f}"
            logging.info(frmt.format(step, logp, -logp,
                                     gp_model.kernel.length_scale.item(),
                                     gp_model.kernel.kernel_scale.item(),
                                     gp_model.obs_noise_scale.item()))

    logging.info("Final exact log prob: {:.6f}".format(-logp))


def main(args):
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    logging.info("Generating data")
    T = args.num_timesteps
    ts = 3.0 * torch.arange(T).float() / T
    ys = torch.sin(2.0 * ts)

    if args.train == "exact":
        exact_run(args, ys)
    else:
        tmc_run(args, ys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cauchy process")
    parser.add_argument("-T", "--num-timesteps", default=10, type=int)
    parser.add_argument("-K", "--num-particles", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--tmc-strategy", default="diagonal", type=str)
    parser.add_argument("--train", default="exact", type=str)
    args = parser.parse_args()
    main(args)
