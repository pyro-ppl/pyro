import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Delta(Distribution):
    """
    Diagonal covariance Normal - the first distribution
    """

    def __init__(self, v, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        if v.dim() == 1 and batch_size > 1:
            self.v = v.expand(v, 0)
        else:
            self.v = v
        super(Delta, self).__init__(*args, **kwargs)

    def sample(self):
        return Variable(self.v)

    def batch_log_pdf(self, x, batch_size=1):
        if x.dim == 1:
            x = x.expand(batch_size, 0)
        return (torch.eq(x, self.v.expand_as(x)) - 1).float() * 999999

    def log_pdf(self, x):
        if torch.equal(x.data, self.v.data.expand_as(x.data)):
            return Variable(torch.zeros(1))
        return Variable(torch.Tensor([-float("inf")]))
