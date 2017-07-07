import pyro
import torch
from queue import Queue

from pyro.infer.trace import Trace
import pyro.poutine as poutine


class TraceSearch(pyro.infer.AbstractInfer):
    """
    New Trace and Poutine-based implementation of systematic search
    """
    def __init__(self, model, queue=None, *args, **kwargs):
        """
        Constructor
        """
        self.model = model
        if queue is None:
            queue = Queue()
            queue.put(Trace())
        self.queue = queue


    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


    def step(self, *args, **kwargs):
        """
        algorithm entered here
        Returns traces from the posterior
        Running until the queue is empty and collecting the marginal histogram
        is performing exact inference
        """
        if not self.queue.empty():
            p = poutine.trace(poutine.beam(self.model, queue=self.queue))
            return p(*args, **kwargs)
        else:
            # the queue is empty - we're done!
            # XXX need to structure this better
            return None
