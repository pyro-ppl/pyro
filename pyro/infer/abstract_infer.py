from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta
from six import add_metaclass

import torch

import pyro.poutine as poutine
from pyro.distributions import Empirical


class EmpiricalMarginal(Empirical):
    """
    :param trace_dist: a TracePosterior instance representing a Monte Carlo posterior.

    Marginal distribution, that wraps over a TracePosterior object to provide a
    a marginal over one or more latent sites or the return values of the
    TracePosterior's model. If multiple sites are specified, they must have the
    same tensor shape.
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
