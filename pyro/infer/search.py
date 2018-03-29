from __future__ import absolute_import, division, print_function

import pyro.poutine as poutine
from pyro.infer import TracePosterior
from six.moves.queue import Queue

from .enum import config_enumerate


class Search(TracePosterior):
    """
    Trace and Poutine-based implementation of systematic search.

    :param callable model: Probabilistic model defined as a function.
    :param int max_tries: The maximum number of times to try completing a trace from the queue.
    """
    def __init__(self, model, max_tries=1e6):
        """
        Constructor. Default `max_tries` to something sensible - 1e6.

        :param callable model: Probabilistic model defined as a function.
        :param int max_tries: The maximum number of times to try completing a trace from the queue.
        """
        self.model = model
        self.max_tries = int(max_tries)

    def _traces(self, *args, **kwargs):
        """
        algorithm entered here
        Running until the queue is empty and collecting the marginal histogram
        is performing exact inference

        :returns: Iterator of traces from the posterior.
        :rtype: Generator[:class:`pyro.Trace`]
        """
        # currently only using the standard library queue
        self.queue = Queue()
        self.queue.put(poutine.Trace())

        p = poutine.trace(
            poutine.queue(self.model, queue=self.queue, max_tries=self.max_tries))
        while not self.queue.empty():
            tr = p.get_trace(*args, **kwargs)
            yield (tr, tr.log_pdf())


class ParallelSearch(TracePosterior):
    """
    TODO docs
    """
    def __init__(self, model, first_available_dim=None):
        """
        TODO docs
        """
        if first_available_dim is None:
            first_available_dim = 0
        self.first_available_dim = first_available_dim
        self.model = model

    def _traces(self, *args, **kwargs):
        """
        TODO docs
        """
        p = poutine.trace(
            poutine.enum(
                config_enumerate(self.model, default="parallel"),
                first_available_dim=self.first_available_dim))
        tr = p.get_trace(*args, **kwargs)
        # now compute joint probabilities:
        # TODO identify all global independence dimensions,
        # and aggregate over all non-global ones
        tr.compute_batch_log_pdf()
        log_joints = sum([tr.nodes[name]["batch_log_pdf"] for name in tr.nodes
                          if tr.nodes[name]["type"] == "sample"])
        yield (tr, log_joints)
