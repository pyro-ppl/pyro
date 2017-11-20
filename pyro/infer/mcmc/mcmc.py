from __future__ import absolute_import, division, print_function

import logging

from pyro.infer import TracePosterior
from pyro import poutine


class MCMC(TracePosterior):
    def __init__(self,
                 model,
                 kernel,
                 warmup_steps,
                 num_samples,
                 *args,
                 **kwargs):
        self.model = model
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.kernel = kernel(model, *args, **kwargs)
        self.logger = logging.getLogger(__name__)
        super(MCMC, self).__init__()

    def _traces(self, *args, **kwargs):
        self.kernel.setup(*args, **kwargs)
        # Run MCMC iterations
        self.t = 0
        trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        while self.t < self.warmup_steps + self.num_samples:
            if self.t % 100 == 0:
                self.logger.info('MCMC Iteration: {}'.format(self.t))
            trace = self.kernel.sample(trace, self.t)
            self.t += 1
            if self.t < self.warmup_steps:
                continue
            yield (trace, trace.log_pdf())
        self.kernel.cleanup()

    @property
    def acceptance_ratio(self):
        return self.kernel.num_accepts / self.t
