import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Uniform(Distribution):
    """
    :param a: lower bound *(real)*
    :param b: upper bound (>a) *(real)*

    Continuous uniform distribution over ``[a, b]``
    """

    def _sanitize_input(self, alpha, beta):
        if alpha is not None:
            # stateless distribution
            return alpha, beta
        elif self.a is not None:
            # stateful distribution
            return self.a, self.b
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, a=None, b=None, *args, **kwargs):
        """
        Params:
          `a` - low bound
          `b` -  high bound
        """
        self.a = a
        self.b = b
        super(Uniform, self).__init__(*args, **kwargs)

    def sample(self, a=None, b=None, *args, **kwargs):
        """
        Reparameterized Uniform sampler.
        """
        _a, _b = self._sanitize_input(a, b)
        eps = Variable(torch.rand(_a.size()).type_as(_a.data))
        return _a + torch.mul(eps, _b - _a)

    def log_pdf(self, x, a=None, b=None, *args, **kwargs):
        """
        Uniform log-likelihood
        """
        _a, _b = self._sanitize_input(a, b)
        if x.dim() == 1:
            if x.le(_a).data[0] or x.ge(_b).data[0]:
                return Variable(torch.Tensor([-float("inf")]).type_as(_a.data))
        else:
            # x is 2-d
            if x.le(_a).data[0, 0] or x.ge(_b).data[0, 0]:
                return Variable(torch.Tensor([[-np.inf]]).type_as(_a.data))
        return torch.sum(-torch.log(_b - _a))

    def batch_log_pdf(self, x, a=None, b=None, batch_size=1, *args, **kwargs):
        _a, _b = self._sanitize_input(a, b)
        if x.dim() == 1 and _a.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, _a, _b)
        _l = x.ge(_a).type_as(_a)
        _u = x.le(_b).type_as(_b)
        return torch.sum(torch.log(_l.mul(_u)) - torch.log(_b - _a), 1)
