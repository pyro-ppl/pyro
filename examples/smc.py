from __future__ import absolute_import, division, print_function

import argparse

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

import torch

# A simple harmonic oscillator with each discretized time step t = 1 
# and mass equal to the spring constant.
#
#   y(t) = (A*z_t + B*eps_t + eps_y)[0]
# 
# where B = [0,0,1], eps_y = [1, 0, 0], eps_t ~ N(0, sigma_eps^2)
# and A is chosen based off Newton's eqns. The prior over sigma is IG(3,1).

class SimpleHarmonicModel:

    def __init__(self, process_noise, measurement_noise):
        self.A = torch.tensor([[0., 1.],
                               [-1., 0.]])
        self.B = torch.tensor([1e-10, 1])
        self.sigma_z = torch.tensor(process_noise)
        self.sigma_y = torch.tensor(measurement_noise)

    def init(self, initial):
        self.t = 0
        self.z = initial
        self.y = None

    def step(self, y=None):
        self.t += 1
        self.z = pyro.sample("z_{}".format(self.t), 
                             dist.Normal(self.z.matmul(self.A), self.B*self.sigma_z).to_event(1))
        self.y = pyro.sample("y_{}".format(self.t), 
                             dist.Normal(self.z[...,0], self.sigma_y),
                             obs = y)

        return self.z, self.y

class SimpleHarmonicModel_Guide:

    def __init__(self, model):
        self.model = model

    def init(self, initial):
        self.t = 0
        self.z = initial

    def step(self, y=None):
        self.t += 1

        # Bad proposal distribution
        self.z = pyro.sample("z_{}".format(self.t), 
                             dist.Normal(self.z.matmul(self.model.A), torch.tensor([1e-10, 2.])).to_event(1))


def _extract_samples(trace):
    return {name: site["value"]
            for name, site in trace.nodes.items()
            if site["type"] == "sample"
            if not site["is_observed"]}


class SMCFilter:
    # TODO: Add window kwarg that defaults to float("inf")
    def __init__(self, model, guide, num_particles, max_plate_nesting):
        self.model = model
        self.guide = guide
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting

        # Equivalent to an empirical distribution.
        self._values = {}
        self._log_weights = torch.zeros(self.num_particles)

    def init(self, *args, **kwargs):
        self.particle_plate = pyro.plate("particles", self.num_particles, dim=-1-self.max_plate_nesting)
        with poutine.block(), self.particle_plate:
            guide_trace = poutine.trace(self.guide.init).get_trace(*args, **kwargs)
            model = poutine.replay(self.model.init, guide_trace)
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)

        self._update_weights(model_trace, guide_trace)
        self._values.update(_extract_samples(model_trace))
        self._maybe_importance_resample()

    def step(self, *args, **kwargs):
        with poutine.block(), self.particle_plate:
            guide_trace = poutine.trace(self.guide.step).get_trace(*args, **kwargs)
            model = poutine.replay(self.model.step, guide_trace)
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)

        self._update_weights(model_trace, guide_trace)
        self._values.update(_extract_samples(model_trace))
        self._maybe_importance_resample()
    
    def get_values_and_log_weights(self):
        return self._values, self._log_weights

    def get_empirical(self):
        return {name: dist.Empirical(value, self._log_weights)
                for name, value in self._values.items()}

    @torch.no_grad()
    def _update_weights(self, model_trace, guide_trace):
        # w_t <-w_{t-1}*p(y_t|z_t) * p(z_t|z_t-1)/q(z_t)

        model_trace.compute_log_prob()
        guide_trace.compute_log_prob()

        for name, guide_site in guide_trace.nodes.items():
            if guide_site["type"] == "sample":
                model_site = model_trace.nodes[name]
                log_p = model_site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                log_q = guide_site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self._log_weights += log_p - log_q

        for site in model_trace.nodes.values():
            if site["type"] == "sample" and site["is_observed"]:
                log_p = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self._log_weights += log_p

        self._log_weights -= self._log_weights.max()

    def _maybe_importance_resample(self):
        if True: # TODO check perplexity
            self._importance_resample()

    def _importance_resample(self):
        # TODO: Turn quadratic algo -> linear algo by being lazier
        index = dist.Categorical(logits=self._log_weights).sample(sample_shape=(self.num_particles,))
        self._values = {name: value[index].contiguous() for name, value in self._values.items()}
        self._log_weights.fill_(0.)


def generate_data(args):
    model = SimpleHarmonicModel(args.process_noise, args.measurement_noise)

    model.init(initial=torch.zeros(2))
    zs = []
    ys = []
    for t in range(args.num_timesteps):
        z, y = model.step()
        zs.append(z)
        ys.append(y)

    return zs, ys


def main(args):
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    model = SimpleHarmonicModel(args.process_noise, args.measurement_noise)
    guide = SimpleHarmonicModel_Guide(model)

    smc = SMCFilter(model, guide, num_particles=args.num_particles, max_plate_nesting=0)

    zs, ys = generate_data(args)
    smc.init(initial=torch.zeros(2))
    for y in ys:
        smc.step(y)

    empirical = smc.get_empirical()
    for t in range(1,args.num_timesteps):
        z = empirical["z_{}".format(t)]
        print("{}\t{}\t{}\t{}".format(t, zs[t], z.mean, z.variance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Harmonic Oscillator w/ SMC")
    parser.add_argument("-n", "--num-timesteps", default=50, type=int)
    parser.add_argument("-p", "--num-particles", default=100, type=int)
    parser.add_argument("--process-noise", default=1., type=float)
    parser.add_argument("--measurement-noise", default=1., type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)

