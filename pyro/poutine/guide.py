# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Union

import torch

import pyro.distributions as dist
from pyro.distributions.distribution import Distribution

from .trace_messenger import TraceMessenger
from .trace_struct import Trace
from .util import prune_subsample_sites, site_is_subsample


class GuideMessenger(TraceMessenger, ABC):
    """
    Abstract base class for effect-based guides.

    Derived classes must implement the :meth:`get_posterior` method.
    """

    def __init__(self, model: Callable):
        super().__init__()
        # Do not register model as submodule
        self._model = (model,)

    @property
    def model(self):
        return self._model[0]

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Draws posterior samples from the guide and replays the model against
        those samples.

        :returns: A dict mapping sample site name to sample value.
            This includes latent, deterministic, and observed values.
        :rtype: dict
        """
        self.args_kwargs = args, kwargs
        try:
            with self:
                self.model(*args, **kwargs)
        finally:
            del self.args_kwargs

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
        posterior = self.get_posterior(msg["name"], prior)
        if isinstance(posterior, torch.Tensor):
            posterior = dist.Delta(posterior, event_dim=prior.event_dim)
        if posterior.batch_shape != prior.batch_shape:
            posterior = posterior.expand(prior.batch_shape)
        msg["fn"] = posterior

    def _pyro_post_sample(self, msg):
        # Manually apply outer plates.
        prior = msg["infer"].get("prior")
        if prior is not None and prior.batch_shape != msg["fn"].batch_shape:
            msg["infer"]["prior"] = prior.expand(msg["fn"].batch_shape)
        return super()._pyro_post_sample(msg)

    @abstractmethod
    def get_posterior(
        self, name: str, prior: Distribution
    ) -> Union[Distribution, torch.Tensor]:
        """
        Abstract method to compute a posterior distribution or sample a
        posterior value given a prior distribution conditioned on upstream
        posterior samples.

        Implementations may use ``pyro.param`` and ``pyro.sample`` inside this
        function, but ``pyro.sample`` statements should set
        ``infer={"is_auxiliary": True"}`` .

        Implementations may access further information for computations:

        - ``value = self.upstream_value(name)`` is the value of an upstream
           sample or deterministic site.
        -  ``self.trace`` is a trace of upstream sites, and may be useful for
           other information such as ``self.trace.nodes["my_site"]["fn"]`` or
           ``self.trace.nodes["my_site"]["cond_indep_stack"]`` .
        -  ``args, kwargs = self.args_kwargs`` are the inputs to the model, and
            may be useful for amortization.

        :param str name: The name of the sample site to sample.
        :param prior: The prior distribution of this sample site
            (conditioned on upstream samples from the posterior).
        :type prior: ~pyro.distributions.Distribution
        :returns: A posterior distribution or sample from the posterior
            distribution.
        :rtype: ~pyro.distributions.Distribution or torch.Tensor
        """
        raise NotImplementedError

    def upstream_value(self, name: str) -> torch.Tensor:
        """
        For use in :meth:`get_posterior` .

        :returns: The value of an upstream sample or deterministic site
        :rtype: torch.Tensor
        """
        return self.trace.nodes[name]["value"]

    def get_traces(self) -> Tuple[Trace, Trace]:
        """
        This can be called after running :meth:`__call__` to extract a pair of
        traces.

        In contrast to the trace-replay pattern of generating a pair of traces,
        :class:`GuideMessenger` interleaves model and guide computations, so
        only a single ``guide(*args, **kwargs)`` call is needed to create both
        traces. This function merely extract the relevant information from this
        guide's ``.trace`` attribute.

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
