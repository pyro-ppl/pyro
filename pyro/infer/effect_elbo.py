# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Tuple, Union

import torch

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.trace_struct import Trace
from pyro.poutine.util import prune_subsample_sites, site_is_subsample
from pyro.util import check_model_guide_match, check_site_shape

from .trace_elbo import JitTrace_ELBO, Trace_ELBO


class GuideMessenger(TraceMessenger, ABC):
    """
    EXPERIMENTAL Abstract base class for effect-based guides for use in
    :class:`Effect_ELBO` and similar.

    Derived classes must implement the :meth:`get_posterior` method.
    """

    def __init__(self, model):
        super().__init__()
        # Do not register model as submodule
        self._model = (model,)

    @property
    def model(self):
        return self._model[0]

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        self.args_kwargs = args, kwargs
        self.upstream_values = OrderedDict()
        try:
            with self:
                self.model(*args, **kwargs)
        finally:
            del self.args_kwargs
            del self.upstream_values

        model_trace, guide_trace = self.get_traces()
        samples = {
            name: site["value"]
            for name, site in model_trace.nodes.items()
            if site["type"] == "sample"
        }
        return samples

    def _pyro_sample(self, msg):
        if msg["is_observed"] or site_is_subsample(msg):
            return
        prior = msg["fn"]
        msg["infer"]["prior"] = prior
        posterior = self.get_posterior(msg["name"], prior, self.upstream_values)
        if isinstance(posterior, torch.Tensor):
            posterior = dist.Delta(posterior, event_dim=prior.event_dim)
        if posterior.batch_shape != prior.batch_shape:
            posterior = posterior.expand(prior.batch_shape)
        msg["fn"] = posterior

    def _pyro_post_sample(self, msg):
        self.upstream_values[msg["name"]] = msg["value"]

        # Manually apply outer plates.
        prior = msg["infer"].get("prior")
        if prior is not None and prior.batch_shape != msg["fn"].batch_shape:
            msg["infer"]["prior"] = prior.expand(msg["fn"].batch_shape)

        return super()._pyro_post_sample(msg)

    @abstractmethod
    def get_posterior(
        self,
        name: str,
        prior: TorchDistribution,
        upstream_values: Dict[str, torch.Tensor],
    ) -> Union[TorchDistribution, torch.Tensor]:
        """
        Abstract method to compute a posterior distribution or sample a
        posterior value given a prior distribution and values of upstream
        sample sites.

        Implementations may use ``pyro.param`` and ``pyro.sample`` inside this
        function, but ``pyro.sample`` statements should set
        ``infer={"is_auxiliary": True"}`` .

        Implementations may access further information for computations:

        -  ``args, kwargs = self.args_kwargs`` are the inputs to the model, and
            may be useful for amortization.
        -  ``self.trace`` is a trace of upstream sites, and may be useful for
           other information such as ``self.trace.nodes["my_site"]["fn"]`` or
           ``self.trace.nodes["my_site"]["cond_indep_stack"]`` .

        :param str name: The name of the sample site to sample.
        :param prior: The prior distribution of this sample site
            (conditioned on upstream samples from the posterior).
        :type prior: ~pyro.distributions.TorchDistribution
        :param dict upstream_values:
        :returns: A posterior distribution or sample from the posterior
            distribution.
        :rtype: ~pyro.distributions.TorchDistribution or torch.Tensor
        """
        raise NotImplementedError

    def get_traces(self) -> Tuple[Trace, Trace]:
        """
        :returns: a pair ``(model_trace, guide_trace)``
        :rtype: tuple
        """
        guide_trace = prune_subsample_sites(self.trace)
        model_trace = model_trace = guide_trace.copy()
        for name, guide_site in list(guide_trace.nodes.items()):
            if guide_site["type"] != "sample" or guide_site["is_observed"]:
                del guide_trace.nodes[name]
                continue
            model_site = model_trace.nodes[name].copy()
            model_site["fn"] = guide_site["infer"]["prior"]
            model_trace.nodes[name] = model_site
        return model_trace, guide_trace


class EffectMixin(ELBO):
    """
    EXPERIMENTAL Mixin class to turn a trace-based ELBO implementation into an
    effect-based implementation.
    """

    def _get_trace(self, model, guide, args, kwargs):
        # This differs from Trace_ELBO in that the guide is assumed to be an
        # effect handler.
        guide(*args, **kwargs)
        while not isinstance(guide, GuideMessenger):
            guide = guide.func.args[1]  # unwrap plates
        model_trace, guide_trace = guide.get_traces()

        # The rest follows pyro.infer.enum.get_importance_trace().
        max_plate_nesting = self.max_plate_nesting
        if is_validation_enabled():
            check_model_guide_match(model_trace, guide_trace, max_plate_nesting)
        model_trace.compute_log_prob()
        guide_trace.compute_score_parts()
        if is_validation_enabled():
            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, max_plate_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, max_plate_nesting)

        return model_trace, guide_trace


class Effect_ELBO(EffectMixin, Trace_ELBO):
    """
    EXPERIMENTAL Similar to :class:`~pyro.infer.trace_elbo.Trace_ELBO` but
    supporting guides that are :class:`GuideMessenger` s rather than traceable
    functions.
    """

    pass


class JitEffect_ELBO(EffectMixin, JitTrace_ELBO):
    """
    EXPERIMENTAL Similar to :class:`~pyro.infer.trace_elbo.JitTrace_ELBO` but
    supporting guides that are :class:`GuideMessenger` s rather than traceable
    functions.
    """

    pass
