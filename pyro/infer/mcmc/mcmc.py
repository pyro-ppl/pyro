from __future__ import absolute_import, division, print_function

import logging
import math

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
    """

    def __init__(self, kernel, num_samples, warmup_steps=0):
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.logger = logging.getLogger(__name__)
        super(MCMC, self).__init__()

    def _traces(self, *args, **kwargs):
        self.kernel.setup(*args, **kwargs)
        trace = self.kernel.initial_trace()
        self.logger.info("Starting MCMC using kernel - {} ...".format(self.kernel.__class__.__name__))
        logging_interval = math.ceil((self.warmup_steps + self.num_samples) / 20)
        for t in range(1, self.warmup_steps + self.num_samples + 1):
            trace = self.kernel.sample(trace)
            if t % logging_interval == 0:
                stage = "WARMUP" if t <= self.warmup_steps else "SAMPLE"
                self.logger.info("Iteration: {} [{}]".format(t, stage))
                diagnostic_info = self.kernel.diagnostics()
                if diagnostic_info is not None:
                    self.logger.info(diagnostic_info)
            if t <= self.warmup_steps:
                if t == self.warmup_steps:
                    self.kernel.end_warmup()
                continue
            yield (trace, 1.0)
        self.kernel.cleanup()
