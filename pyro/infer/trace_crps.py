import operator
from collections import OrderedDict
from functools import reduce

import torch

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import scale_and_mask
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled
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


class Trace_CRPS:
    """
    Posterior predictive CRPS or energy loss [1] with optional ``KL(q,p)``
    regularization.

    This is a likelihood-free method, and can be used for likelihoods without
    tractible density functions. CRPS is a robust loss function, and is well
    defined for any distribution with finite absolute moment ``E[|data|]``.

    This requires static model structure, a fully reparametrized guide, and
    reparametrized likelihood distributions in the model. Model latent
    distributions may be non-reparametrized.

    References
    [1] `Strictly Proper Scoring Rules, Prediction, and Estimation`
    Tilmann Gneiting, Adrian E. Raftery (2007)
    https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param int num_particles: The number of particles/samples used to form the
        gradient estimators. Must be at least 2.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. If omitted, this will guess a valid value
        by running the (model,guide) pair once.
    :param float kl_scale: Nonnegative scale for ``KL(q,p)`` regularization.
        If zero (default), then log densities will not be computed.
    :param float beta: Exponent ``beta`` from [1]. The loss function is
        strictly proper for distributions with finite ``beta``-absolute moment
        ``E[||X||^beta]``; thus for heavy tailed distributions, ``beta`` should
        be small, e.g. for ``Cauchy`` distributions, ``beta<1`` is strictly
        proper.  Defaults to 1. Must be in the open interval (0,2).
    :param float tol: Small nonnegative number added to squared norms to
        stabilize gradients. The loss function is strictly proper even for
        large values of ``tol``.
    """
    def __init__(self,
                 num_particles=2,
                 max_plate_nesting=float('inf'),
                 kl_scale=0.,
                 beta=1.,
                 tol=0.1):
        if not (isinstance(num_particles, int) and num_particles >= 2):
            raise ValueError("Expected num_particles >= 2, actual {}".format(num_particles))
        if not (isinstance(kl_scale, (float, int)) and kl_scale >= 0):
            raise ValueError("Expected kl_scale >= 0, actual {}".format(kl_scale))
        if not (isinstance(beta, (float, int)) and 0 < beta and beta < 2):
            raise ValueError("Expected beta in (0,2), actual {}".format(beta))
        if not (isinstance(tol, (float, int)) and 0 <= tol):
            raise ValueError("Expected tol >= 0, actual {}".format(tol))
        self.num_particles = num_particles
        self.vectorize_particles = True
        self.max_plate_nesting = max_plate_nesting
        self.kl_scale = kl_scale
        self.beta = beta
        self.tol = tol

    def _pow(self, x):
        # Numerically stabilize to avoid nan grads near zero.
        # By Thereom 5 of (Gneiting and Raftery 2007), the stabilized loss
        # remains strictly proper even for large values of tol.
        x = x.add(self.tol ** 2)

        if self.beta == 1:
            return x.sqrt()  # cheaper than .pow()
        return x.pow(self.beta / 2)

    def _get_traces(self, model, guide, *args, **kwargs):
        if self.max_plate_nesting == float("inf"):
            # TODO factor this out as a stand-alone helper.
            ELBO._guess_max_plate_nesting(self, model, guide, *args, **kwargs)
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
                        raise ValueError("Trace_CRPS requires fully reparametrized guides")
            for trace in model_trace.nodes.values():
                if site["type"] == "sample":
                    if site["is_observed"]:
                        warn_if_nan(site["value"], site["name"])
                        if not getattr(site["fn"], "has_rsample", False):
                            raise ValueError("Trace_CRPS requires reparametrized likelihoods")

        if self.kl_scale > 0:
            guide_trace.compute_log_prob()
            model_trace.compute_log_prob(site_filter=lambda name, site: not site["is_observed"])
            if is_validation_enabled():
                for site in guide_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_plate_nesting)
                for trace in model_trace.nodes.values():
                    if site["type"] == "sample":
                        if not site["is_observed"]:
                            check_site_shape(site, self.max_plate_nesting)

        return guide_trace, model_trace

    def __call__(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters.
        """
        guide_trace, model_trace = self._get_traces(model, guide, *args, **kwargs)

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

        # Compute crps from mean average error and generalized entropy.
        squared_error = []  # E[ (X - x)^2 ]
        squared_entropy = []  # E[ (X - X')^2 ]
        prototype = next(iter(data.values()))
        pairs = prototype.new_ones(self.num_particles, self.num_particles).tril(-1).nonzero()
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
        crps = error - 0.5 * entropy

        # Compute KL(guide,model).
        kl_qp = 0
        if self.kl_scale > 0:
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    kl_qp = kl_qp + site["log_prob_sum"]
            for site in model_trace.nodes.values():
                if site["type"] == "sample" and not site["is_observed"]:
                    kl_qp = kl_qp - site["log_prob_sum"]

        # Compute final loss.
        loss = crps + self.kl_scale * kl_qp
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        Not implemented. Added for compatibility with unit tests only.
        """
        raise NotImplementedError("Trace_CRPS implements only surrogate loss")
