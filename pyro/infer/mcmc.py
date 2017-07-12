import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.distributions import Uniform, Categorical


class MCMC(pyro.infer.abstract_infer.AbstractInfer):
    """
    Initial implementation of MCMC
    """
    def __init__(self, model, guide=None, proposal=None, samples=10, lag=1, burn=0):
        super(MCMC, self).__init__()
        self.samples = samples
        self.lag = lag
        self.burn = burn
        self.model = model
        assert (guide is None or proposal is None) and \
            (guide is not None or proposal is not None), \
            "cannot have guide and proposal"
        if guide is not None:
            self.guide = lambda tr, *args, **kwargs: guide(*args, **kwargs)
        else:
            self.guide = proposal

    def _dist(self, *args, **kwargs):
        """
        make trace posterior distribution
        """
        # initialize traces with a draw from the prior
        old_model_trace = poutine.trace(self.model)(*args, **kwargs)
        traces = []
        t = 0
        while t < self.burn + self.lag * self.samples:
            # p(x, z)
            old_model_trace = traces[-1]
            # q(z' | z)
            new_guide_trace = poutine.block(
                poutine.trace(self.guide))(old_model_trace, *args, **kwargs)
            # p(x, z')
            new_model_trace = poutine.trace(
                poutine.replay(self.model, new_guide_trace))(*args, **kwargs)
            # q(z | z')
            old_guide_trace = poutine.block(
                poutine.trace(
                    poutine.replay(self.guide, new_model_trace)))(new_model_trace,
                                                                  *args, **kwargs)
            # p(x, z') q(z' | z) / p(x, z) q(z | z')
            logr = new_model_trace.log_pdf() + new_guide_trace.log_pdf() - \
                   old_model_trace.log_pdf() + old_guide_trace.log_pdf()
            rnd = pyro.sample("mh_step_{}".format(i),
                              Uniform(pyro.zeros(1), pyro.ones(1)))
            if torch.log(rnd)[0] < logr[0]:
                # accept
                old_model_trace = new_model_trace
                if t <= self.burn or (t > self.burn and t % self.lag == 0):
                    t += 1
                    traces.append(new_model_trace)

        trace_ps = Variable(torch.Tensor([tr.log_pdf() for tr in traces]))
        trace_ps -= pyro.util.log_sum_exp(trace_ps)
        return Categorical(ps=torch.exp(trace_ps), vs=traces)

    def sample(self, *args, **kwargs):
        """
        sample from trace posterior
        """
        return self._dist(*args, **kwargs).sample()

    def log_pdf(self, val, *args, **kwargs):
        return self._dist(*args, **kwargs).log_pdf(val)

    def log_z(self, *args, **kwargs):
        traces = self._dist(*args, **kwargs).vs
        log_z = 0.0
        for tr in traces:
            log_z = log_z + tr.log_pdf()
        return log_z / len(traces)
