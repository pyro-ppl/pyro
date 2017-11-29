from __future__ import absolute_import, division, print_function

import logging

from pyro.infer import TracePosterior
from pyro import poutine


class MCMC(TracePosterior):
    """
    Wrapper class for Markov Chain Monte Carlo algorithms. Specific MCMC algorithms
    are TraceKernel instances and need to be supplied as a ``kernel`` argument
    to the constructor.

    :param model: A stochastic function containing Pyro primitive calls.
    :param kernel: An implementation of the ``TraceKernel`` class, which when
        given an execution trace returns another sample trace from the target
        (posterior) distribution.
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int warmup_steps: Number of warmup iterations. The samples generated
        during the warmup phase are discarded.
    :param \*args: Positional arguments for constructing the ``kernel`` instance.
    :param \*\*kwargs: Arbitrary keyword arguments for constructing the ``kernel``
        instance.
    """

    def __init__(self, model, kernel, warmup_steps, num_samples, *args, **kwargs):
        self.model = model
        self.kernel = kernel(model, *args, **kwargs)
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        if warmup_steps >= num_samples:
            raise ValueError('Number of warmup iterations - {} >= Number of MCMC samples - {}'
                             .format(warmup_steps, num_samples))
        self._t = None
        self.logger = logging.getLogger(__name__)
        super(MCMC, self).__init__()

    def _traces(self, *args, **kwargs):
        self.kernel.setup(*args, **kwargs)
        self._t = 0
        trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        self.logger.info('Starting MCMC using kernel - {} ...'.format(self.kernel.__class__.__name__))
        logging_interval = int((self.warmup_steps + self.num_samples) / 20)
        while self._t < self.warmup_steps + self.num_samples:
            if self._t % logging_interval == 0:
                self.logger.info('MCMC iteration: {}.'.format(self._t))
                if self.kernel.diagnostics is not None:
                    self.logger.info(self.kernel.diagnostics(self._t))
            trace = self.kernel.sample(trace, self._t, *args, **kwargs)
            self._t += 1
            if self._t < self.warmup_steps:
                continue
            yield (trace, 1.0)
        self.kernel.cleanup()
