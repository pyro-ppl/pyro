import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine

class SGMCMC(pyro.infer.abstract_infer.AbstractInfer):
    """
    sketch of stochastic gradient MCMC
    """
    def __init__(self, model, optimizer, samples=10, lag=1, burn=0):
        super(SGMCMC, self).__init__()
        self.samples = samples
        self.lag = lag
        self.burn = burn
        self.model = model
        self.optimizer = optimizer

    def _traces(self, *args, **kwargs):
        """
        main control loop
        """
        traces = []
        tr = poutine.trace(self.model)(*args, **kwargs)
        for i in range(num_samples):
            tr = poutine.trace(poutine.replay(self.model, tr))(*args, **kwargs)
            logp = tr.log_pdf()
            tr_samples = [tr[name]["value"] for name in tr.keys() \
                          if tr[name]["type"] == "param"]
            autograd.backward(tr_samples, logp)
            self.optimizer.step(tr_samples)
            zero_grad(tr_samples)
            traces.append(tr.copy())

        log_weights = [tr.log_pdf() for tr in traces]
        return traces, log_weights
