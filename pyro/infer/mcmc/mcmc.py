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
        self._t = None
        # Initialize logger
        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        super(MCMC, self).__init__()

    def _traces(self, *args, **kwargs):
        self.kernel.setup(*args, **kwargs)
        self._t = 0
        trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        self.logger.info('Starting MCMC using kernel - {}'.format(self.kernel.__class__.__name__))
        while self._t < self.warmup_steps + self.num_samples:
            if self._t % 100 == 0:
                self.logger.info('MCMC Iteration: {}'.format(self._t))
            trace = self.kernel.sample(trace, self._t)
            self._t += 1
            if self._t < self.warmup_steps:
                continue
            yield (trace, trace.log_pdf())
        self.kernel.cleanup()

    @property
    def acceptance_ratio(self):
        return self.kernel.num_accepts / self._t
