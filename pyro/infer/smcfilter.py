from __future__ import absolute_import, division, print_function

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

import torch


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
        if True:  # TODO check perplexity
            self._importance_resample()

    def _importance_resample(self):
        # TODO: Turn quadratic algo -> linear algo by being lazier
        index = dist.Categorical(logits=self._log_weights).sample(sample_shape=(self.num_particles,))
        self._values = {name: value[index].contiguous() for name, value in self._values.items()}
        self._log_weights.fill_(0.)
