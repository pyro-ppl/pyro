from __future__ import absolute_import, division, print_function

import math

import pyro
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
        super(MCMC, self).__init__()

    def _traces(self, *args, **kwargs):
        self.kernel.setup(self.warmup_steps, *args, **kwargs)
        trace = self.kernel.initial_trace()
        pyro.log.info("Starting MCMC using kernel - {} ...".format(self.kernel.__class__.__name__))
        logging_interval = math.ceil((self.warmup_steps + self.num_samples) / 20)
        for t in range(1, self.warmup_steps + self.num_samples + 1):
            trace = self.kernel.sample(trace)
            if t % logging_interval == 0:
                stage = "WARMUP" if t <= self.warmup_steps else "SAMPLE"
                pyro.log.info("Iteration: {} [{}]".format(t, stage))
                diagnostic_info = self.kernel.diagnostics()
                if diagnostic_info is not None:
                    pyro.log.info(diagnostic_info)
            if t <= self.warmup_steps:
                continue
            yield (trace, 1.0)
        self.kernel.cleanup()
