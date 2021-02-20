# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import torch

import pyro.distributions as dist
from pyro.distributions.util import sum_leftmost
from pyro import poutine
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample


def _make_cls(base, static_attrs, instance_attrs, parent_linkage=None):
    r"""
    Dynamically create classes named `_ + base.__name__`, which extend the
    base class with other optional instance and class attributes, and have
    a custom `.expand` method to propagate these attributes on expanded
    instances.

    :param cls base: Base class.
    :param dict static_attrs: static attributes to add to class.
    :param dict instance_attrs: instance attributes for initialization.
    :param str parent_linkage: attribute in the parent class that holds
        a reference to the distribution class.
    :return cls: dynamically generated class.
    """
    def _expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(cls, _instance)
        for attr in instance_attrs:
            setattr(new, attr, getattr(self, attr))
        if parent_linkage:
            setattr(new.parent, parent_linkage, new)
        return base.expand(self, batch_shape, _instance=new)

    name = "_" + base.__name__
    cls = type(name, (base,), instance_attrs)
    for k, v in static_attrs.items():
        setattr(cls, k, v)
    cls.expand = _expand
    return cls


def _latent(base, parent):
    return _make_cls(base, {"collapsible": True}, {"site_name": None, "parent": parent}, "_latent")


def _conditional(base, parent):
    return _make_cls(base, {"marginalize_latent": True}, {"parent": parent}, "_conditional")


def _compound(base, parent):
    return _make_cls(base, {}, {"parent": parent})


class BetaBinomialPair:
    def __init__(self):
        self._latent = None
        self._conditional = None

    def latent(self, *args, **kwargs):
        self._latent = _latent(dist.Beta, parent=self)(*args, **kwargs)
        return self._latent

    def conditional(self, *args, **kwargs):
        self._conditional = _conditional(dist.Binomial, parent=self)(*args, **kwargs)
        return self._conditional

    def posterior(self, obs):
        concentration1 = self._latent.concentration1
        concentration0 = self._latent.concentration0
        total_count = self._conditional.total_count
        reduce_dims = len(obs.size()) - len(concentration1.size())
        # Unexpand total_count to have the same shape as concentration0.
        # Raise exception if this isn't possible.
        total_count = sum_leftmost(total_count, reduce_dims)
        summed_obs = sum_leftmost(obs, reduce_dims)
        return dist.Beta(concentration1 + summed_obs,
                         total_count + concentration0 - summed_obs,
                         validate_args=self._latent._validate_args)

    def compound(self):
        return _compound(dist.BetaBinomial, parent=self)(concentration1=self._latent.concentration1,
                                                         concentration0=self._latent.concentration0,
                                                         total_count=self._conditional.total_count)


class GammaPoissonPair:
    def __init__(self):
        self._latent = None
        self._conditional = None

    def latent(self, *args, **kwargs):
        self._latent = _latent(dist.Gamma, parent=self)(*args, **kwargs)
        return self._latent

    def conditional(self, *args, **kwargs):
        self._conditional = _conditional(dist.Poisson, parent=self)(*args, **kwargs)
        return self._conditional

    def posterior(self, obs):
        concentration = self._latent.concentration
        rate = self._latent.rate
        reduce_dims = len(obs.size()) - len(rate.size())
        num_obs = obs.shape[:reduce_dims].numel()
        summed_obs = sum_leftmost(obs, reduce_dims)
        return dist.Gamma(concentration + summed_obs, rate + num_obs)

    def compound(self):
        return _compound(dist.GammaPoisson, parent=self)(concentration=self._latent.concentration,
                                                         rate=self._latent.rate)


class UncollapseConjugateMessenger(Messenger):
    r"""
    Replay regular sample sites in addition to uncollapsing any collapsed
    conjugate sites.
    """
    def __init__(self, trace):
        """
        :param trace: a trace whose values should be reused

        Constructor.
        Stores trace in an attribute.
        """
        self.trace = trace
        super().__init__()

    def _pyro_sample(self, msg):
        is_collapsible = getattr(msg["fn"], "collapsible", False)
        # uncollapse conjugate sites.
        if is_collapsible:
            conj_node, parent = None, None
            for site_name in self.trace.observation_nodes + self.trace.stochastic_nodes:
                parent = getattr(self.trace.nodes[site_name]["fn"], "parent", None)
                if parent is not None and parent._latent.site_name == msg["name"]:
                    conj_node = self.trace.nodes[site_name]
                    break
            assert conj_node is not None, "Collapsible latent site `{}` with no corresponding conjugate site."\
                .format(msg["name"])
            msg["fn"] = parent.posterior(conj_node["value"])
            msg["value"] = msg["fn"].sample()
        # regular replay behavior.
        else:
            name = msg["name"]
            if name in self.trace:
                guide_msg = self.trace.nodes[name]
                if msg["is_observed"]:
                    return None
                if guide_msg["type"] != "sample":
                    raise RuntimeError("site {} must be sample in trace".format(name))
                msg["done"] = True
                msg["value"] = guide_msg["value"]
                msg["infer"] = guide_msg["infer"]


def uncollapse_conjugate(fn=None, trace=None):
    r"""
    This is similar to :function:`~pyro.poutine.replay` poutine, but in addition to
    replaying the values at sample sites from the ``trace`` in the original callable
    ``fn`` when the same sites are sampled, this also "uncollapses" any observed
    compound distributions (defined in :module:`pyro.distributions.conjugate`)
    by sampling the originally collapsed parameter values from its posterior distribution
    followed by observing the data with the sampled parameter values.
    """
    msngr = UncollapseConjugateMessenger(trace)
    return msngr(fn) if fn is not None else msngr


class CollapseConjugateMessenger(Messenger):
    def _pyro_sample(self, msg):
        is_collapsible = getattr(msg["fn"], "collapsible", False)
        marginalize_latent = getattr(msg["fn"], "marginalize_latent", False)
        if is_collapsible:
            msg["fn"].site_name = msg["name"]
            msg["stop"] = True
        elif marginalize_latent:
            msg["fn"] = msg["fn"].parent.compound()
        else:
            return


def collapse_conjugate(fn=None):
    r"""
    This replaces a latent-observed pair by collapsing the latent site
    (whose distribution has attribute `collapsible=True`), and replacing the
    observed site (whose distribution has attribute `marginalize_latent=True`)
    with a compound probability distribution that marginalizes out the latent
    site.
    """
    msngr = CollapseConjugateMessenger()
    return msngr(fn) if fn is not None else msngr


def posterior_replay(model, posterior_samples, *args, **kwargs):
    r"""
    Given a model and samples from the posterior (potentially with conjugate sites
    collapsed), return a `dict` of samples from the posterior with conjugate sites
    uncollapsed. Note that this can also be used to generate samples from the
    posterior predictive distribution.

    :param model: Python callable.
    :param dict posterior_samples: posterior samples keyed by site name.
    :param args: arguments to `model`.
    :param kwargs: keyword arguments to `model`.
    :return: `dict` of samples from the posterior.
    """
    posterior_samples = posterior_samples.copy()
    num_samples = kwargs.pop("num_samples", None)
    assert posterior_samples or num_samples, "`num_samples` must be provided if `posterior_samples` is empty."
    if num_samples is None:
        num_samples = list(posterior_samples.values())[0].shape[0]

    return_samples = defaultdict(list)
    for i in range(num_samples):
        conditioned_nodes = {k: v[i] for k, v in posterior_samples.items()}
        collapsed_trace = poutine.trace(poutine.condition(collapse_conjugate(model), conditioned_nodes))\
            .get_trace(*args, **kwargs)
        trace = poutine.trace(uncollapse_conjugate(model, collapsed_trace)).get_trace(*args, **kwargs)
        for name, site in trace.iter_stochastic_nodes():
            if not site_is_subsample(site):
                return_samples[name].append(site["value"])

    return {k: torch.stack(v) for k, v in return_samples.items()}
