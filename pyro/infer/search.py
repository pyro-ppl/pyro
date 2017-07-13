import pyro
import torch
import sys
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

import pyro.poutine as poutine


class Search(pyro.infer.abstract_infer.AbstractInfer):
    """
    New Trace and Poutine-based implementation of systematic search
    """
    def __init__(self, model, queue=None, max_tries=1e6):
        """
        Constructor
        """
        self.model = model
        if queue is None:
            queue = Queue()
            queue.put(poutine.Trace())
        self.queue = queue
        self.max_tries = int(max_tries)

    def _traces(self, *args, **kwargs):
        """
        algorithm entered here
        Returns traces from the posterior
        Running until the queue is empty and collecting the marginal histogram
        is performing exact inference
        """
        p = poutine.queue(poutine.trace(self.model),
                          queue=self.queue,
                          max_tries=self.max_tries)
        traces = []
        while not self.queue.empty():
            traces.append(p(*args, **kwargs))

        log_weights = [tr.log_pdf() for tr in traces]
        return traces, log_weights
