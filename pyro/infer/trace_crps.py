import operator
from collections import OrderedDict
from functools import reduce

import torch

import pyro
import pyrp.poutine as poutine
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
    Posterior predictive CRPS loss.

    This is a likelihood-free method; no densities are evaluated.

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators. Must be at least 2.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is only required when enumerating
        over sample sites in parallel, e.g. if a site sets
        ``infer={"enumerate": "parallel"}``. If omitted, ELBO may guess a valid
        value by running the (model,guide) pair once, however this guess may
        be incorrect if model or guide structure is dynamic.
    :param bool retain_graph: Whether to retain autograd graph during an SVI
        step. Defaults to None (False).
    """
    def __init__(self,
                 num_particles=2,
                 max_plate_nesting=float('inf'),
                 retain_graph=None):
        assert isinstance(num_particles, int) and num_particles >= 2
        self.max_plate_nesting = max_plate_nesting
        self.vectorize_particles = True
        self.retain_graph = retain_graph

    def _get_traces(self, model, guide, *args, **kwargs):
        if self._max_plate_nesting == float("inf"):
            ELBO._guess_max_plate_nesting(self, model, guide, *args, **kwargs)
        vectorize = pyro.plate("particles", self.num_particles,
                               dim=-1 - self.max_plate_nesting)

        # Trace the guide as in ELBO.
        with poutine.trace() as tr, vectorize:
            guide(*args, **kwargs)
        guide_trace = tr.trace

        # Trace the model, saving obs in tr2 and posterior predictives in tr1.
        with poutine.trace() as tr1, poutine.uncondition():
            with poutine.trace() as tr2:
                with poutine.replay(trace=guide_trace), vectorize:
                    model(*args, **kwargs)
        pred_trace = tr1.trace
        obs_trace = tr2.trace

        if is_validation_enabled():
            check_model_guide_match(pred_trace, guide_trace, self.max_plate_nesting)
            check_model_guide_match(obs_trace, guide_trace, self.max_plate_nesting)

        guide_trace = prune_subsample_sites(guide_trace)
        pred_trace = prune_subsample_sites(pred_trace)
        obs_trace = prune_subsample_sites(obs_trace)
        if self.kl_scale > 0:
            obs_trace.compute_log_prob(site_filter=lambda name, site:
                                       not site["is_observed"] and site["mask"] is not False)

        if is_validation_enabled():
            for trace in [guide_trace, pred_trace, obs_trace]:
                for site in trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_plate_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    if not getattr(site["fn"], "has_rsample", False):
                        raise ValueError("Trace_CRPS only supports fully reparametrized guides")

        return guide_trace, obs_trace, pred_trace

    def __call__(self, model, guide, *args, **kwargs):
        guide_trace, obs_trace, pred_trace = self._get_traces(model, guide, *args, **kwargs)

        # Extract observations and posterior predictive samples.
        data = OrderedDict()
        samples = OrderedDict()
        for name, site in obs_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                data[name] = site["value"]
                samples[name] = pred_trace.nodes[name]["value"]
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
            scale = obs_trace.nodes["scale"]
            mask = obs_trace.nodes["mask"]

            # Flatten.
            batch_shape = obs.shape[:obs.dim() - obs_trace.nodes[name]["fn"].event_dim]
            if isinstance(scale, torch.Tensor):
                scale = scale.expand_as(batch_shape).reshape(-1)
            if isinstance(mask, torch.Tensor):
                mask = mask.expand_as(batch_shape).reshape(-1)
            obs = obs.reshape(-1)
            sample = sample.reshape(self.num_samples, -1)

            squared_error.append(_squared_error(sample, obs, scale, mask))
            squared_entropy.append(_squared_error(*sample[pairs], scale, mask))

        squared_error = reduce(operator.add, squared_error)
        squared_entropy = reduce(operator.add, squared_entropy)
        error = squared_error.sqrt().mean()  # E[ |X-x| ]
        entropy = squared_entropy.sqrt().mean()  # E[ |X-X'| ]
        crps = error - 0.5 * entropy

        # Compute log p(z).
        logp = 0
        if self.kl_scale > 0:
            for site in obs_trace.nodes.values():
                if "log_prob" in site:
                    logp = logp + site["log_prob"]

        # Compute final loss.
        loss = crps - logp
        warn_if_nan(loss, "loss")
        return loss
