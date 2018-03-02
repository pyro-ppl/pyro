from __future__ import absolute_import, division, print_function

import pyro.poutine as poutine
from pyro.infer import TracePosterior
from six.moves.queue import Queue


class Search(TracePosterior):
    """
    Trace and Poutine-based implementation of systematic search.

    :param callable model: Probabilistic model defined as a function.
    :param int max_tries: The maximum number of times to try completing a trace from the queue.
    """
    def __init__(self, model, max_tries=int(1e6)):
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

        p = poutine.trace(poutine.queue(self.model, queue=self.queue))

        tries = 0
        while not self.queue.empty():
            tries += 1
            if tries >= self.max_tries:
                raise ValueError("max tries ({}) exceeded".format(self.max_tries))

            tr = p.get_trace(*args, **kwargs)
            yield (tr, tr.log_pdf())
