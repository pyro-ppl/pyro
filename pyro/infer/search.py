import pyro
import torch
import sys
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

from pyro.poutine import Trace
import pyro.poutine as poutine


class Search(pyro.infer.abstract_infer.AbstractInfer):
    """
    New Trace and Poutine-based implementation of systematic search
    """
    def __init__(self, model, queue=None, **kwargs):
        """
        Constructor
        """
        self.model = model
        if queue is None:
            queue = Queue()
            queue.put(Trace())
        self.queue = queue

    def runner(self, num_samples, *args, **kwargs):
        """
        algorithm entered here
        Returns traces from the posterior
        Running until the queue is empty and collecting the marginal histogram
        is performing exact inference
        """
        p = poutine.queue(poutine.trace(self.model),
                          queue=self.queue,
                          max_tries=num_samples)
        samples = []
        for i in range(num_samples):
            if self.queue.empty():
                break
            model_trace = p(*args, **kwargs)
            samples.append([i, model_trace["_RETURN"]["value"], model_trace.log_pdf()])
        return samples
