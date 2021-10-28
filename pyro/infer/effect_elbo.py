# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, Tuple, Union

import torch

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.trace_struct import TraceStruct
from pyro.poutine.util import prune_subsample_sites, site_is_subsample
from pyro.util import check_model_guide_match, check_site_shape

from .trace_elbo import JitTrace_ELBO, Trace_ELBO


class GuideMessengerMeta(type(TraceMessenger), ABCMeta):
    pass


class GuideMessenger(TraceMessenger, metaclass=GuideMessengerMeta):
    """
    Abstract base class for effect-based guides for use in :class:`Effect_ELBO`
    .

    Derived classes must implement the :meth:`get_posterior` method.
    """

    def __enter__(self, *args, **kwargs) -> TraceStruct:
        self.args_kwargs = args, kwargs
        self.upstream_values = OrderedDict()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.args_kwargs
        del self.upstream_values
        return super().__exit__(self, exc_type, exc_value, traceback)

    def _pyro_sample(self, msg):
        if not msg["is_observed"] and not site_is_subsample(msg):
            msg["infer"]["prior"] = msg["fn"]
            posterior = self.get_posterior(msg["name"], msg["fn"], self.upstream_values)
            if isinstance(posterior, torch.Tensor):
                posterior = dist.Delta(posterior, event_dim=msg["fn"].event_dim)
            msg["fn"] = posterior
        return super()._pyro_sample(msg)

    def _pyro_post_sample(self, msg):
        self.upstream_values[msg["name"]] = msg["value"]
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

    def get_traces(self) -> Tuple[TraceStruct, TraceStruct]:
        """
        :returns: a pair ``(model_trace, guide_trace)``
        :rtype: tuple
        """
        guide_trace = self.trace.copy()
        model_trace = self.trace.copy()
        for name, guide_site in list(guide_trace.nodes.items()):
            if guide_site["type"] != "sample" or guide_site["is_observed"]:
                del guide_trace.nodes[name]
                continue
            model_trace[name]["fn"] = guide_site["infer"]["prior"]
        return model_trace, guide_trace


class EffectMixin(ELBO):
    """
    Mixin class to turn a trace-based ELBO implementation into an effect-based
    implementation.
    """

    def _get_trace(
        self, model: Callable, guide: GuideMessenger, args: tuple, kwargs: dict
    ):
        # This differs from Trace_ELBO in that the guide is assumed to be an
        # effect handler.
        assert isinstance(guide, GuideMessenger)
        with guide(*args, **kwargs):
            model(*args, **kwargs)
        model_trace, guide_trace = guide.get_traces()
        if getattr(self, "max_plate_nesting") is None:
            self.max_plate_nesting = max(
                [0]
                + [
                    -f.dim
                    for site in guide.trace.nodes.values()
                    for f in site["cond_indep_stack"]
                    if f.vectorized
                ]
            )

        # The rest follows pyro.infer.enum.get_importance_trace().
        max_plate_nesting = self.max_plate_nesting
        if is_validation_enabled():
            check_model_guide_match(model_trace, guide_trace, max_plate_nesting)
        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)
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
    Similar to :class:`~pyro.infer.trace_elbo.Trace_ELBO` but supporting guides
    that are effect handlers rather than traceable functions.
    """

    pass


class JitEffect_ELBO(EffectMixin, JitTrace_ELBO):
    """
    Similar to :class:`~pyro.infer.trace_elbo.JitTrace_ELBO` but supporting guides
    that are effect handlers rather than traceable functions.
    """

    pass
