import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Delta(Distribution):
    """
    Delta Distribution - probability of 1 at `v`
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
        if isinstance(self.v, Variable):
            return self.v
        return Variable(self.v)

    def batch_log_pdf(self, x, batch_size=1):
        if x.dim == 1:
            x = x.expand(batch_size, 0)
        return (torch.eq(x, self.v.expand_as(x)) - 1).float() * 999999

    def log_pdf(self, x):
        if torch.equal(x.data, self.v.data.expand_as(x.data)):
            return Variable(torch.zeros(1))
        return Variable(torch.Tensor([-float("inf")]))

    def support(self):
        # univariate case
        return (Variable(self.v.data.index(i)) for i in range(self.v.size(0)))
