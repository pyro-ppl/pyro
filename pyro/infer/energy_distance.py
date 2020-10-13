# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import operator
from collections import OrderedDict
from functools import reduce

import torch

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import scale_and_mask
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled, validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, warn_if_nan


def _squared_error(x, y, scale, mask):
    diff = x - y
    if getattr(scale, 'shape', ()) or getattr(mask, 'shape', ()):
        error = torch.einsum("nbe,nbe->nb", diff, diff)
        return scale_and_mask(error, scale, mask).sum(-1)
    else:
        error = torch.einsum("nbe,nbe->n", diff, diff)
        return scale_and_mask(error, scale, mask)


class EnergyDistance:
    r"""
    Posterior predictive energy distance [1,2] with optional Bayesian
    regularization by the prior.

    Let `p(x,z)=p(z) p(x|z)` be the model, `q(z|x)` be the guide.  Then given
    data `x` and drawing an iid pair of samples :math:`(Z,X)` and
    :math:`(Z',X')` (where `Z` is latent and `X` is the posterior predictive),

    .. math ::

        & Z \sim q(z|x); \quad X \sim p(x|Z) \\
        & Z' \sim q(z|x); \quad X' \sim p(x|Z') \\
        & loss = \mathbb E_X \|X-x\|^\beta
               - \frac 1 2 \mathbb E_{X,X'}\|X-X'\|^\beta
               - \lambda \mathbb E_Z \log p(Z)

    This is a likelihood-free inference algorithm, and can be used for
    likelihoods without tractable density functions. The :math:`\beta` energy
    distance is a robust loss functions, and is well defined for any
    distribution with finite fractional moment :math:`\mathbb E[\|X\|^\beta]`.

    This requires static model structure, a fully reparametrized guide, and
    reparametrized likelihood distributions in the model. Model latent
    distributions may be non-reparametrized.

    **References**

    [1] Gabor J. Szekely, Maria L. Rizzo (2003)
        Energy Statistics: A Class of Statistics Based on Distances.
    [2] Tilmann Gneiting, Adrian E. Raftery (2007)
        Strictly Proper Scoring Rules, Prediction, and Estimation.
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param float beta: Exponent :math:`\beta` from [1,2]. The loss function is
        strictly proper for distributions with finite :math:`beta`-absolute moment
        :math:`E[\|X\|^\beta]`. Thus for heavy tailed distributions ``beta`` should
        be small, e.g. for ``Cauchy`` distributions, :math:`\beta<1` is strictly
        proper. Defaults to 1. Must be in the open interval (0,2).
    :param float prior_scale: Nonnegative scale for prior regularization.
        Model parameters are trained only if this is positive.
        If zero (default), then model log densities will not be computed
        (guide log densities are never computed).
    :param int num_particles: The number of particles/samples used to form the
        gradient estimators. Must be at least 2.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. If omitted, this will guess a valid value
        by running the (model,guide) pair once.
    """
    def __init__(self,
                 beta=1.,
                 prior_scale=0.,
                 num_particles=2,
                 max_plate_nesting=float('inf')):
        if not (isinstance(beta, (float, int)) and 0 < beta and beta < 2):
            raise ValueError("Expected beta in (0,2), actual {}".format(beta))
        if not (isinstance(prior_scale, (float, int)) and prior_scale >= 0):
            raise ValueError("Expected prior_scale >= 0, actual {}".format(prior_scale))
        if not (isinstance(num_particles, int) and num_particles >= 2):
            raise ValueError("Expected num_particles >= 2, actual {}".format(num_particles))
        self.beta = beta
        self.prior_scale = prior_scale
        self.num_particles = num_particles
        self.vectorize_particles = True
        self.max_plate_nesting = max_plate_nesting

    def _pow(self, x):
        if self.beta == 1:
            return x.sqrt()  # cheaper than .pow()
        return x.pow(self.beta / 2)

    def _get_traces(self, model, guide, args, kwargs):
        if self.max_plate_nesting == float("inf"):
            with validation_enabled(False):  # Avoid calling .log_prob() when undefined.
                # TODO factor this out as a stand-alone helper.
                ELBO._guess_max_plate_nesting(self, model, guide, args, kwargs)
        vectorize = pyro.plate("num_particles_vectorized", self.num_particles,
                               dim=-self.max_plate_nesting)

        # Trace the guide as in ELBO.
        with poutine.trace() as tr, vectorize:
            guide(*args, **kwargs)
        guide_trace = tr.trace

        # Trace the model, drawing posterior predictive samples.
        with poutine.trace() as tr, poutine.uncondition():
            with poutine.replay(trace=guide_trace), vectorize:
                model(*args, **kwargs)
        model_trace = tr.trace
        for site in model_trace.nodes.values():
            if site["type"] == "sample" and site["infer"].get("was_observed", False):
                site["is_observed"] = True
        if is_validation_enabled():
            check_model_guide_match(model_trace, guide_trace, self.max_plate_nesting)

        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)
        if is_validation_enabled():
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    warn_if_nan(site["value"], site["name"])
                    if not getattr(site["fn"], "has_rsample", False):
                        raise ValueError("EnergyDistance requires fully reparametrized guides")
            for trace in model_trace.nodes.values():
                if site["type"] == "sample":
                    if site["is_observed"]:
                        warn_if_nan(site["value"], site["name"])
                        if not getattr(site["fn"], "has_rsample", False):
                            raise ValueError("EnergyDistance requires reparametrized likelihoods")

        if self.prior_scale > 0:
            model_trace.compute_log_prob(site_filter=lambda name, site: not site["is_observed"])
            if is_validation_enabled():
                for site in model_trace.nodes.values():
                    if site["type"] == "sample":
                        if not site["is_observed"]:
                            check_site_shape(site, self.max_plate_nesting)

        return guide_trace, model_trace

    def __call__(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters.
        """
        guide_trace, model_trace = self._get_traces(model, guide, args, kwargs)

        # Extract observations and posterior predictive samples.
        data = OrderedDict()
        samples = OrderedDict()
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                data[name] = site["infer"]["obs"]
                samples[name] = site["value"]
        assert list(data.keys()) == list(samples.keys())
        if not data:
            raise ValueError("Found no observations")

        # Compute energy distance from mean average error and generalized entropy.
        squared_error = []  # E[ (X - x)^2 ]
        squared_entropy = []  # E[ (X - X')^2 ]
        prototype = next(iter(data.values()))
        pairs = prototype.new_ones(self.num_particles, self.num_particles).tril(-1).nonzero(as_tuple=False)
        for name, obs in data.items():
            sample = samples[name]
            scale = model_trace.nodes[name]["scale"]
            mask = model_trace.nodes[name]["mask"]

            # Flatten to subshapes of (num_particles, batch_size, event_size).
            event_dim = model_trace.nodes[name]["fn"].event_dim
            batch_shape = obs.shape[:obs.dim() - event_dim]
            event_shape = obs.shape[obs.dim() - event_dim:]
            if getattr(scale, 'shape', ()):
                scale = scale.expand(batch_shape).reshape(-1)
            if getattr(mask, 'shape', ()):
                mask = mask.expand(batch_shape).reshape(-1)
            obs = obs.reshape(batch_shape.numel(), event_shape.numel())
            sample = sample.reshape(self.num_particles, batch_shape.numel(), event_shape.numel())

            squared_error.append(_squared_error(sample, obs, scale, mask))
            squared_entropy.append(_squared_error(*sample[pairs].unbind(1), scale, mask))

        squared_error = reduce(operator.add, squared_error)
        squared_entropy = reduce(operator.add, squared_entropy)
        error = self._pow(squared_error).mean()  # E[ ||X-x||^beta ]
        entropy = self._pow(squared_entropy).mean()  # E[ ||X-X'||^beta ]
        energy = error - 0.5 * entropy

        # Compute prior.
        log_prior = 0
        if self.prior_scale > 0:
            for site in model_trace.nodes.values():
                if site["type"] == "sample" and not site["is_observed"]:
                    log_prior = log_prior + site["log_prob_sum"]

        # Compute final loss.
        loss = energy - self.prior_scale * log_prior
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        Not implemented. Added for compatibility with unit tests only.
        """
        raise NotImplementedError("EnergyDistance implements only surrogate loss")
