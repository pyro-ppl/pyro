import pyro
import torch
import sys
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

from pyro.poutine import Trace
import pyro.poutine as poutine
from pyro.distributions import Categorical


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
            queue.put(Trace())
        self.queue = queue
        self.max_tries = int(max_tries)

    def _dist(self, *args, **kwargs):
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

        log_ps = Variable(torch.Tensor([tr.log_pdf() for tr in traces]))
        log_ps = log_ps - pyro.util.log_sum_exp(log_ps)
        return Categorical(ps=torch.exp(log_ps), vs=traces)

    def sample(self, *args, **kwargs):
        """
        sample from trace posterior
        """
        return self._dist(*args, **kwargs).sample()

    def log_pdf(self, val, *args, **kwargs):
        return self._dist(*args, **kwargs).log_pdf(val)
