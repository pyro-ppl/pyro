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
    error = torch.einsum("np,np->n", diff, diff)
    return scale_and_mask(error, scale, mask)


class Trace_CRPS:
    """
    Posterior predictive CRPS loss with ``KL(q,p)`` regularization.

    This is a likelihood-free method, and can be used for likelihoods without
    tractible density functions. CRPS is a robust loss function, and is well
    defined for any distribution with finite absolute moment ``E[|data|]``.

    This requires static model structure, fully reparametrized guide, and
    reparametrized likelihood distributions in the model. Model latent
    distributions may be non-reparametrized.

    Note that in the loss ``CRPS + KL(q,p)``, the ``CRPS`` term has data units
    whereas the ``KL(q,p)`` term has units of nats.  To calibrate these units,
    you can wrap likelihood sites in ``poutine.scale``.

    References
    [1] `Strictly Proper Scoring Rules, Prediction, and Estimation`
    Tilmann Gneiting, Adrian E. Raftery (2007)
    https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param int num_particles: The number of particles/samples used to form the
        gradient estimators. Must be at least 2.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is only required when enumerating
        over sample sites in parallel, e.g. if a site sets
        ``infer={"enumerate": "parallel"}``. If omitted, ELBO may guess a valid
        value by running the (model,guide) pair once, however this guess may
        be incorrect if model or guide structure is dynamic.
    """
    def __init__(self,
                 num_particles=2,
                 max_plate_nesting=float('inf')):
        if not isinstance(num_particles, int) and num_particles >= 2:
            raise ValueError("Expected num_particles >= 2, actual {}".format(num_particles))
        self.num_particles = num_particles
        self.vectorize_particles = True
        self.max_plate_nesting = max_plate_nesting

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
        guide_trace.compute_log_prob()
        model_trace.compute_log_prob(site_filter=lambda name, site: not site["is_observed"])
        if is_validation_enabled():
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_plate_nesting)
                    if not getattr(site["fn"], "has_rsample", False):
                        raise ValueError("Trace_CRPS requires fully reparametrized guides")
            for trace in model_trace.nodes.values():
                if site["type"] == "sample":
                    if site["is_observed"]:
                        if not getattr(site["fn"], "has_rsample", False):
                            raise ValueError("Trace_CRPS requires reparametrized likelihoods")
                    else:
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

            # Flatten.
            batch_shape = obs.shape[:obs.dim() - model_trace.nodes[name]["fn"].event_dim]
            if isinstance(scale, torch.Tensor):
                scale = scale.expand(batch_shape).reshape(-1)
            if isinstance(mask, torch.Tensor):
                mask = mask.expand(batch_shape).reshape(-1)
            obs = obs.reshape(-1)
            sample = sample.reshape(self.num_particles, -1)

            squared_error.append(_squared_error(sample, obs, scale, mask))
            squared_entropy.append(_squared_error(*sample[pairs].unbind(1), scale, mask))

        squared_error = reduce(operator.add, squared_error)
        squared_entropy = reduce(operator.add, squared_entropy)
        error = squared_error.sqrt().mean()  # E[ |X-x| ]
        entropy = squared_entropy.sqrt().mean()  # E[ |X-X'| ]
        crps = error - 0.5 * entropy

        # Compute KL(guide||model).
        kl_qp = 0
        for site in model_trace.nodes.values():
            if site["type"] == "sample" and not site["is_observed"]:
                kl_qp = kl_qp + site["log_prob_sum"]
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                kl_qp = kl_qp - site["log_prob_sum"]

        # Compute final loss.
        loss = crps + kl_qp
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs):
        """
        Not implemented. Added for compatibility with unit tests only.
        """
        raise NotImplementedError("Trace_CRPS implements only surrogate loss")
