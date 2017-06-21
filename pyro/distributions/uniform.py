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
        eps = Variable(torch.rand(self.a.size()),
                       requires_grad=False).type_as(self.mu)
        return self.a + torch.mul(eps, torch.Tensor.sub(self.b, self.a))

    def log_pdf(self, x):
        """
        Normal log-likelihood
        """
        return torch.sum(-torch.log(self.b - self.a))

    def batch_log_pdf(self, x, batch_size=1):
        raise NotImplementedError()

    def support(self, x):
        return {'lower': self.a, 'upper': self.b}
