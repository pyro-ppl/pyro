from __future__ import absolute_import, division, print_function

import logging
import math

import torch
from torch.autograd import Variable

from pyro.infer import TracePosterior


class MCMC(TracePosterior):
    """
    Wrapper class for Markov Chain Monte Carlo algorithms. Specific MCMC algorithms
    are TraceKernel instances and need to be supplied as a ``kernel`` argument
    to the constructor.

    :param kernel: An instance of the ``TraceKernel`` class, which when
        given an execution trace returns another sample trace from the target
        (posterior) distribution.
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int warmup_steps: Number of warmup iterations. The samples generated
        during the warmup phase are discarded.
    :param int sample_every: The number of steps between each trace acceptance
        after the warmup phase. Sometimes described as lag.
    """

    def __init__(self, kernel, num_samples, warmup_steps=0, sample_every=1):
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.sample_every = sample_every
        if warmup_steps >= num_samples:
            raise ValueError('Number of warmup iterations - {} >= Number of MCMC samples - {}'
                             .format(warmup_steps, num_samples))
        self._t = None
        self.logger = logging.getLogger(__name__)
        super(MCMC, self).__init__()

    def _traces(self, *args, **kwargs):
        self.kernel.setup(*args, **kwargs)
        self._t = 0
        trace = self.kernel.initial_trace()
        self.logger.info('Starting MCMC using kernel - {} ...'.format(self.kernel.__class__.__name__))
        logging_interval = int(math.ceil((self.warmup_steps + self.num_samples) / 20))
        while self._t <= self.warmup_steps + self.num_samples:
            if self._t % logging_interval == 0:
                self.logger.info('Iteration: {}.'.format(self._t))
                diagnostic_info = self.kernel.diagnostics(self._t)
                if diagnostic_info is not None:
                    self.logger.info(diagnostic_info)
            trace = self.kernel.sample(trace)

            self._t += 1
            if self._t <= self.warmup_steps or self._t % self.sample_every != 0:
                continue
            yield (trace, Variable(torch.Tensor([1.0])))

        self.kernel.cleanup()
