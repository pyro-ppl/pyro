from __future__ import absolute_import, division, print_function

import json
import logging

from pyro.infer import TracePosterior
from pyro.infer.mcmc.logger import initialize_logger, initialize_progbar, TQDM_MSG


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
        self.logger = logging.getLogger("pyro.infer.mcmc")
        super(MCMC, self).__init__()

    def _gen_samples(self, num_samples, init_trace):
        trace = init_trace
        for _ in range(num_samples):
            trace = self.kernel.sample(trace)
            diagnostics = json.dumps(self.kernel.diagnostics())
            self.logger.info(diagnostics, extra={"msg_type": TQDM_MSG})
            yield trace

    def _traces(self, *args, **kwargs):
        chain_id = kwargs.pop("chain_id", 0)
        progress_bar = initialize_progbar(self.warmup_steps, self.num_samples)
        self.logger = initialize_logger(self.logger, chain_id, progress_bar)
        self.kernel.setup(*args, **kwargs)
        trace = self.kernel.initial_trace()
        with progress_bar:
            for trace in self._gen_samples(self.warmup_steps, trace):
                continue
            self.kernel.end_warmup()
            if progress_bar:
                progress_bar.set_description("Sample")
            for trace in self._gen_samples(self.num_samples, trace):
                yield (trace, 1.0)
        self.kernel.cleanup()
