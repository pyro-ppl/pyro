# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functorch.dim import dims

import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites


def log_density(fn, args, kwargs):
    """
    Compute log density of a stochastic function given its arguments.

    :param fn: Python callable containing Pyro primitives.
    :param tuple args: args provided to the function.
    :param dict kwargs: kwargs provided to the function.
    :return: log of joint density and a corresponding model trace
    """
    fn_trace = prune_subsample_sites(poutine.trace(fn).get_trace(*args, **kwargs))
    log_joint = 0.0
    for site in fn_trace.nodes.values():
        if site["type"] == "sample" and site["fn"]:
            value = site["value"]
            scale = site["scale"]
            log_prob = site["fn"].log_prob(value)

            if scale is not None:
                log_prob = scale * log_prob

            sum_dims = tuple(f.dim for f in site["cond_indep_stack"])
            log_joint += log_prob.sum(sum_dims)
    return log_joint, fn_trace


class ELBO:
    def __init__(self, num_particles=1, vectorize_particles=True):
        self.num_particles = num_particles
        self.vectorize_particles = vectorize_particles

    def loss(self, model, guide, *args, **kwargs):
        if self.num_particles > 1:
            vectorize = pyro.plate("num_particles", self.num_particles, dim=dims(1))
            model = vectorize(model)
            guide = vectorize(guide)

        guide_log_density, guide_trace = log_density(guide, args, kwargs)
        replay_model = poutine.replay(model, trace=guide_trace)
        model_log_density, model_trace = log_density(replay_model, args, kwargs)
        elbo = (model_log_density - guide_log_density) / self.num_particles
        return -elbo
