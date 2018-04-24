from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import torch
from six import add_metaclass

import pyro.poutine as poutine
from pyro.distributions import Categorical, Empirical


class EmpiricalMarginal(Empirical):
    """
    Marginal distribution, that wraps over a TracePosterior object to provide a
    a marginal over one or more latent sites or the return values of the
    TracePosterior's model. If multiple sites are specified, they must have the
    same tensor shape.

    :param TracePosterior trace_posterior: a TracePosterior instance representing
        a Monte Carlo posterior.
    :param list sites: optional list of sites for which we need to generate
        the marginal distribution for. Note that for multiple sites, the shape
        for the site values must match.
    """
    def __init__(self, trace_posterior, sites=None, validate_args=None):
        assert isinstance(trace_posterior, TracePosterior), \
            "trace_dist must be trace posterior distribution object"
        super(EmpiricalMarginal, self).__init__(validate_args=validate_args)
        if sites is None:
            sites = "_RETURN"
        self._populate_traces(trace_posterior, sites)

    def _populate_traces(self, trace_posterior, sites):
        assert isinstance(sites, (list, str))
        for tr, log_weight in zip(trace_posterior.exec_traces, trace_posterior.log_weights):
            value = tr.nodes[sites]["value"] if isinstance(sites, str) else \
                torch.stack([tr.nodes[site]["value"] for site in sites], 0)
            self.add(value, log_weight=log_weight)


@add_metaclass(ABCMeta)
class TracePosterior(object):
    """
    Abstract TracePosterior object from which posterior inference algorithms inherit.
    When run, collects a bag of execution traces from the approximate posterior.
    This is designed to be used by other utility classes like `EmpiricalMarginal`,
    that need access to the collected execution traces.
    """
    def __init__(self):
        self._init()

    def _init(self):
        self.log_weights = []
        self.exec_traces = []

    @abstractmethod
    def _traces(self, *args, **kwargs):
        """
        Abstract method implemented by classes that inherit from `TracePosterior`.

        :return: Generator over ``(exec_trace, weight)``.
        """
        raise NotImplementedError("inference algorithm must implement _traces")

    def run(self, *args, **kwargs):
        """
        Calls `self._traces` to populate execution traces from a stochastic
        Pyro model.

        :param args: optional args taken by `self._traces`.
        :param kwargs: optional keywords args taken by `self._traces`.
        """
        self._init()
        for tr, logit in poutine.block(self._traces)(*args, **kwargs):
            self.exec_traces.append(tr)
            self.log_weights.append(logit)
        return self


class PosteriorPredictive(TracePosterior):
    """
    Generates and holds traces from the posterior predictive distribution,
    given model execution traces from the approximate posterior.

    :param model: arbitrary Python callable containing Pyro primitives.
    :param model_traces: execution traces from the model.
    :param int num_samples: number of samples to generate.
    :param list hide_nodes: list of nodes that should be hidden when
        replaying the model against ``model_traces``. Usually, this
        corresponds to observed nodes.
    :param list log_weights: optional importance weights of execution
        traces to be used during sampling.
    """
    def __init__(self, model, model_traces, num_samples, hide_nodes=[], log_weights=None):
        self.model = model
        self.model_traces = model_traces
        self.hide_nodes = hide_nodes
        self.num_samples = num_samples
        if not log_weights:
            log_weights = torch.zeros(len(model_traces,))
        else:
            log_weights = torch.tensor(log_weights)
        self._categorical = Categorical(logits=log_weights)
        super(PosteriorPredictive, self).__init__()

    def _get_random_pruned_trace(self):
        random_idx = self._categorical.sample()
        trace = self.model_traces[random_idx].copy()
        for node in self.hide_nodes:
            trace.remove_node(node)
        return trace

    def _traces(self, *args, **kwargs):
        for _ in range(self.num_samples):
            model_trace = self._get_random_pruned_trace()
            replayed_trace = poutine.trace(poutine.replay(self.model, model_trace)).get_trace(*args, **kwargs)
            yield (replayed_trace, 0)
