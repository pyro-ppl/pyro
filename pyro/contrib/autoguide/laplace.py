from __future__ import absolute_import, division, print_function

import torch

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.autoguide import AutoDelta


def _hessian(y, xs):
    dys = torch.autograd.grad(y, xs, create_graph=True)
    flat_dy = torch.cat([dy.reshape(-1) for dy in dys])
    H = []
    for dyi in flat_dy:
        Hi = torch.cat([Hij.reshape(-1) for Hij in torch.autograd.grad(dyi, xs, retain_graph=True)])
        H.append(Hi)
    H = torch.stack(H)
    return H


class AutoLaplace(AutoDelta):
    """
    Laplace approximation (quadratic approximation) approximates the posterior
    :math:`log p(z | x)` by a multivariate normal distribution. Under the hood,
    it uses Delta distributions to construct a MAP guide over the entire latent
    space. Its covariance is given by the inverse of the hessian of
    :math:`-\log p(x, z)` at the MAP point of `z`.

    .. note:: The support of posterior is always real.

    Usage::

        guide = AutoLaplace(model)
        svi = SVI(model, guide, ...)
        posterior = guide.get_posterior()

    By default latent variables are randomly initialized by the model. To
    change this default behavior the user should call :func:`pyro.param` before
    beginning inference, with ``"auto_"`` prefixed to the targetd sample site
    names e.g. for sample sites named "level" and "concentration", initialize
    via::

        pyro.param("auto_level", torch.tensor([-1., 0., 1.]))
        pyro.param("auto_concentration", torch.ones(k),
                   constraint=constraints.positive)
    """

    def get_posterior(self, *args, **kwargs):
        guide_trace = poutine.trace(self).get_trace(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(self.model, trace=guide_trace)).get_trace(*args, **kwargs)
        loss = - model_trace.log_prob_sum()

        latents = []
        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                latents.append(site[name]["value"])

        flat_latent = torch.cat([latent.reshape(-1) for latent in latents])
        H = _hessian(loss, latents)
        return dist.MultivariateNormal(loc=flat_latent, precision_matrix=H)
