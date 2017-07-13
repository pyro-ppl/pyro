import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Uniform(Distribution):
    """
    Diagonal covariance Normal - the first distribution
    """

    def __init__(self, a, b, *args, **kwargs):
        """
        * `low = a`,
        * `high = b`,
        """
        self.a = a
        self.b = b
        super(Uniform, self).__init__(*args, **kwargs)

    def sample(self):
        """
        Reparametrized Uniform sampler.
        """
        eps = Variable(torch.rand(self.a.size()))
        return self.a + torch.mul(eps, torch.Tensor.sub(self.b, self.a))

    def log_pdf(self, x):
        """
        Normal log-likelihood
        """
        if x.dim() == 1:
            if x.le(self.a).data[0] or x.ge(self.b).data[0]:
                return Variable(torch.Tensor([-float("inf")]))
        else:
            # x is 2-d
            if x.le(self.a).data[0, 0] or x.ge(self.b).data[0, 0]:
                return Variable(torch.Tensor([[-float("inf")]]))
        return torch.sum(-torch.log(self.b - self.a))

    def batch_log_pdf(self, x, batch_size=1):
        if x.dim() == 1 and self.a.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        _l = x.ge(self.a).type_as(self.a)
        _u = x.le(self.b).type_as(self.b)
        return torch.sum(torch.log(_l.mul(_u)) - torch.log(self.b - self.a), 1)

    def support(self):
        raise NotImplementedError("Support not supported for continuous distributions")
