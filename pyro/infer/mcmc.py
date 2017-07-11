import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.distributions import Uniform


class MCMC(pyro.infer.abstract_infer.AbstractInfer):
    """
    Initial implementation of MCMC
    """
    def __init__(self, model, guide=None, proposal=None):
        super(MCMC, self).__init__()
        self.model = model
        assert (guide is None or proposal is None) and \
            (guide is not None or proposal is not None), \
            "cannot have guide and proposal"
        if guide is not None:
            self.guide = lambda tr, *args, **kwargs: guide(*args, **kwargs)
        else:
            self.guide = proposal

    def runner(self, num_samples, *args, **kwargs):
        """
        main control loop
        """
        # initialize traces with a draw from the prior
        traces = [poutine.trace(self.model)(*args, **kwargs)]
        for i in range(num_samples):
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
                traces.append(new_model_trace)

        samples = [[i, tr["_RETURN"]["value"], tr.log_pdf()] for tr in traces]
        return samples
