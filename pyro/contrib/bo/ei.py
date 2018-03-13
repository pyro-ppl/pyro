import numpy as np
# from scipy.stats import norm
from .bo import BO
from torch.distributions import Normal
import torch
class EI(BO):
    def __init__(self, X, y, kernel, noise=None):
        super().__init__(X, y, kernel, noise=noise)
        # loc = torch.zeros_like(X[0,:])
        # scale = torch.ones_like(X[0,:])
        # print(f'loc : {loc.shape}, scale : {scale.shape}')
        # self.norm = Normal(loc, scale)

    
    def acquire(self, x, xi=0.0):
        """
        Expected Improvement acquisition function

        :param torch.autograd.Variable x: A 1D or 2D tensor of inputs.
        :param torch.autograd.Variable xi: hyperparameter to choose between exploitation and exploration
        """
        m, std = self.predict(x, return_std=True)
        norm = Normal(torch.zeros_like(m), torch.ones_like(m))
        # print(f'Acq m : {m.shape}')
        z = (m - self.y.max() - xi)/std
        f = (z * norm.cdf(z) + torch.exp(norm.log_prob(z))) * std
        return f


