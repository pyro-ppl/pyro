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
    reparameterized = True

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
        a, b = self._sanitize_input(a, b)
        eps = Variable(torch.rand(a.size()).type_as(a.data))
        return a + torch.mul(eps, b - a)

    def log_pdf(self, x, a=None, b=None, *args, **kwargs):
        """
        Uniform log-likelihood
        """
        a, b = self._sanitize_input(a, b)
        if x.dim() == 1:
            if x.le(a).data[0] or x.ge(b).data[0]:
                return Variable(torch.Tensor([-float("inf")]).type_as(a.data))
        else:
            # x is 2-d
            if x.le(a).data[0, 0] or x.ge(b).data[0, 0]:
                return Variable(torch.Tensor([[-np.inf]]).type_as(a.data))
        return torch.sum(-torch.log(b - a))

    def batch_log_pdf(self, x, a=None, b=None, batch_size=1, *args, **kwargs):
        a, b = self._sanitize_input(a, b)
        assert a.dim() == b.dim()
        if x.dim() == 1 and a.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, a, b)
        l = x.ge(a).type_as(a)
        u = x.le(b).type_as(b)
        return torch.sum(torch.log(l.mul(u)) - torch.log(b - a), 1)

    def analytic_mean(self, a=None, b=None):
        a, b = self._sanitize_input(a, b)
        return 0.5 * (a + b)

    def analytic_var(self, a=None, b=None):
        a, b = self._sanitize_input(a, b)
        return torch.pow(b - a, 2) / 12
