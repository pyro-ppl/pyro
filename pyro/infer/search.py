import pyro
import torch
import sys
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

import pyro.poutine as poutine
from pyro.infer import TracePosterior


class Search(TracePosterior):
    """
    New Trace and Poutine-based implementation of systematic search
    """
    def __init__(self, model, queue=None, max_tries=1e6):
        """
        Constructor
        """
        self.model = model
        # XXX add queue here or on call to _traces?
        if queue is None:
            queue = Queue()
        self.queue = queue
        self.max_tries = int(max_tries)

    def _traces(self, *args, **kwargs):
        """
        algorithm entered here
        Returns traces from the posterior
        Running until the queue is empty and collecting the marginal histogram
        is performing exact inference
        """
        if self.queue.empty():
            self.queue.put(poutine.Trace())

        p = poutine.trace(
            poutine.queue(self.model, queue=self.queue, max_tries=self.max_tries))
        while not self.queue.empty():
            tr = p(*args, **kwargs)
            yield (tr, tr.log_pdf())
