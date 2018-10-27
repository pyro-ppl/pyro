from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import itertools

import torch
from opt_einsum import shared_intermediates
from torch.distributions.utils import broadcast_all

import pyro
import pyro.distributions as dist
from pyro.distributions.util import logsumexp

import pyro.poutine as poutine
from pyro.ops.contract import contract_to_tensor

from .traceenum_elbo import _make_dist


class CollapseEnumMessenger(poutine.enumerate_messenger.EnumerateMessenger):

    def _pyro_sample(self, msg):
        collapse = msg["infer"].get("collapse")
        if collapse:
            msg["infer"]["enumerate"] = "parallel"

        super(CollapseEnumMessenger, self)._pyro_sample(msg)

        if collapse:
            msg["is_observed"] = True


def _is_collapsed(site):
    # TODO fix behavior around replay
    return site["type"] == "sample" and site["infer"].get("collapse")


class CollapseSampleMessenger(pyro.poutine.messenger.Messenger):
    """
    Implements forward filtering / backward sampling for sampling
    from the joint posterior distribution
    """
    def __init__(self, enum_trace):
        self.enum_trace = enum_trace
        self.enum_trace.compute_log_prob()
        self.cache = None
        self.enum_dims = set()

        for name, site in self.enum_trace.nodes.items():
            if _is_collapsed(site):
                self.enum_dims.add(site["fn"].event_dim - site["value"].dim())

        self.log_probs = {}
        for name, site in self.enum_trace.nodes.items():
            if site["type"] == "sample":
                log_prob = site["log_prob"]
                for dim in range(-log_prob.dim(), 0):
                    if dim in self.enum_dims:
                        log_prob = logsumexp(log_prob, dim, keepdim=True)
                self.log_probs[name] = log_prob

        self.log_prob_sum = sum(log_prob.sum() for log_prob in self.log_probs.values())

    def __enter__(self):

        self.log_factors = OrderedDict()
        for site in self.enum_trace.nodes.values():
            if site["type"] != "sample":
                continue
            ordinal = frozenset(f for f in site["cond_indep_stack"] if f.vectorized)
            self.log_factors.setdefault(ordinal, []).append(site["log_prob"])

        self.sum_dims = {x: set(i for i in range(-x.dim(), 0) if x.shape[i] > 1) & self.enum_dims
                         for xs in self.log_factors.values() for x in xs}

        self.cache = {}
        return super(CollapseSampleMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        assert not any(self.sum_dims.values())
        for i in itertools.count():
            name = "_collapse_{}".format(i)
            if name not in self.enum_trace:
                break
        # TODO put this into the right plate (the intersection of collapsed variable plates)
        # used for e.g. vectorized num_particles
        pyro.sample(name, dist.Bernoulli(logits=self.log_prob_sum), obs=torch.tensor(1.),
                    infer={"is_auxiliary": True})
        return super(CollapseSampleMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        name = msg["name"]
        if name in self.log_probs:
            msg["log_prob"] = self.log_probs[name]

        if not _is_collapsed(msg):
            # return super(CollapseSampleMessenger, self)._process_message(msg)
            return None

        enum_dim = self.enum_trace.nodes[name]["infer"].get("_enumerate_dim")
        assert enum_dim is not None

        msg["infer"]["_enumerate_dim"] = enum_dim
        assert enum_dim < 0, "{} {}".format(name, enum_dim)
        for value in self.sum_dims.values():
            value.discard(enum_dim)
        with shared_intermediates(self.cache) as cache:
            ordinal = frozenset(f for f in msg["cond_indep_stack"] if f.vectorized)
            logits = contract_to_tensor(self.log_factors, self.sum_dims, ordinal, cache=cache)
            logits = logits.unsqueeze(-1).transpose(-1, enum_dim - 1)
            while logits.shape[0] == 1:
                logits.squeeze_(0)

        msg["fn"] = _make_dist(msg["fn"], logits)
        msg["stop"] = True

    def _postprocess_message(self, msg):
        if not _is_collapsed(msg):
            return super(CollapseSampleMessenger, self)._postprocess_message(msg)

        enum_dim = msg["infer"].get("_enumerate_dim")
        if enum_dim is not None:
            for t, terms in self.log_factors.items():
                for i, term in enumerate(terms):
                    if term.dim() >= -enum_dim and term.shape[enum_dim] > 1:
                        term_, value_ = broadcast_all(term, msg["value"])
                        value_ = value_.index_select(enum_dim, value_.new_tensor([0], dtype=torch.long))
                        sampled_term = term_.gather(enum_dim, value_.long())
                        terms[i] = sampled_term
                        self.sum_dims[sampled_term] = self.sum_dims.pop(term) - {enum_dim}


def collapse(model, first_available_dim):
    """
    Use `ubersum` to collapse sample sites marked with `site["infer"]["collapse"] = True`

    .. warning:: Cannot be wrapped with :func:~`pyro.poutine.replay`
    """

    def _collapsed_model(*args, **kwargs):
        with poutine.block():
            enum_trace = poutine.trace(
                CollapseEnumMessenger(first_available_dim)(model)
            ).get_trace(*args, **kwargs)

        with CollapseSampleMessenger(enum_trace):
            return model(*args, **kwargs)

    return _collapsed_model
