import torch
from functorch.dim import dims

import pyro
from pyro import poutine
from pyro.distributions.util import is_identically_zero


def log_density(model, args, kwargs):
    """
    (EXPERIMENTAL INTERFACE) Computes log of joint density for the model given
    latent values ``params``.

    :param model: Python callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :return: log of joint density and a corresponding model trace
    """
    model_trace = poutine.trace(model).get_trace(*args, **kwargs)
    log_joint = 0.0
    for site in model_trace.nodes.values():
        if site["type"] == "sample" and site["fn"]:
            value = site["value"]
            scale = site["scale"]
            log_prob = site["fn"].log_prob(value)

            if scale is not None:
                log_prob = scale * log_prob

            sum_dims = getattr(log_prob, "dims", ()) + tuple(range(log_prob.ndim))
            log_prob = log_prob.sum(sum_dims)
            log_joint = log_joint + log_prob
    return log_joint, model_trace


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
