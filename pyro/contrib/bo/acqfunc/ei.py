import numpy as np
# from scipy.stats import norm
from .bo import BO
from torch.distributions import Normal
import torch
from .acq_func import AcquisitionFunction

class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp):
        self.gp = gp


    def __call__(self, x, xi=0.0):
        m, cov = self.gp(x, full_cov=True)
        std = cov.diag().sqrt()
        norm = Normal(torch.zeros_like(m), torch.ones_like(m))
        z = (m - self.gp.y.max() - xi)/std
        f = (z * norm.cdf(z) + torch.exp(norm.log_prob(z))) * std
        return f
