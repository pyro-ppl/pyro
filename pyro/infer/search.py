import pyro
import torch
import sys
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

from pyro.poutine import Trace
import pyro.poutine as poutine


class Search(pyro.infer.AbstractInfer):
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
        """
        Really need to work on the inference interface
        """
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        algorithm entered here
        Returns traces from the posterior
        Running until the queue is empty and collecting the marginal histogram
        is performing exact inference
        """
        if not self.queue.empty():
            p = poutine.trace(poutine.queue(self.model, queue=self.queue))
            return p(*args, **kwargs)
        else:
            # the queue is empty - we're done!
            # XXX need to structure this better
            return None
