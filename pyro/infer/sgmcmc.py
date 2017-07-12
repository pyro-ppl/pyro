import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine

class SGMCMC(pyro.infer.abstract_infer.AbstractInfer):
    """
    sketch of stochastic gradient MCMC
    """
    def __init__(self, model):
        self.model = model

    def runner(self, num_samples, *args, **kwargs):
        """
        main control loop
        """
        samples = []
        tr = poutine.trace(self.model)(*args, **kwargs)
        for i in range(num_samples):
            tr = poutine.trace(poutine.replay(self.model, tr))(*args, **kwargs)
            logp = tr.log_pdf()
            tr_samples = tr.filter(site_type="sample")
            autograd.backward(tr_samples, logp)
            optimizer.step(tr_samples)
            zero_grad(tr_samples)
            samples.append([i, tr.copy(True)["_RETURN"]["value"], logp])
        return samples
