import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Bernoulli(Distribution):
    """
    :param ps: probabilities *(vector [0, 1])*

    Distribution over a vector of independent Bernoulli variables. Each element
    of the vector takes on a value in ``{0, 1}``.
    """
    enumerable = True

    def __init__(self, ps=None, is_log_prob=False):
        """
        :param ps: tensor of probabilities or log probabilities
        :param is_log_prob: determines whether `ps` should be interpreted as
            log probabilities.
        """
        self.ps = ps
        self.is_log_prob = is_log_prob
        super(Bernoulli, self).__init__()

    def batch_shape(self, x=None):
        event_dim = 1
        ps = self.ps
        if x is not None and x.size() != self.ps.size():
            ps = self.ps.expand_as(x)
        return ps.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        return Variable(torch.bernoulli(self.ps.data))

    def batch_log_pdf(self, x, log_pdf_mask=None):
        ps = self.ps
        ps = ps.expand(self.shape(x))
        x_1 = x - 1
        ps_1 = ps - 1
        x = x.type_as(ps)
        x_1 = x_1.type_as(x_1)
        xmul = torch.mul(x, ps)
        xmul_1 = torch.mul(x_1, ps_1)
        logsum = torch.log(torch.add(xmul, xmul_1))

        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if log_pdf_mask is not None:
            logsum = logsum * log_pdf_mask
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return torch.sum(logsum, -1).contiguous().view(batch_log_pdf_shape)

    def support(self):
        """
        Returns the Bernoulli distribution's support, as a tensor along the first dimension.

        Note that this returns support values of all the batched RVs in lock-step, rather
        than the full cartesian product. To iterate over the cartesian product, you must
        construct univariate Bernoullis and use itertools.product() over all univariate
        variables (may be expensive).

        :return: torch variable enumerating the support of the Bernoulli distribution.
            Each item in the return value, when enumerated along the first dimensions, yields a
            value from the distribution's support which has the same dimension as would be returned by
            sample.
        :rtype: torch.autograd.Variable.
        """
        return Variable(torch.stack([torch.Tensor([t]).expand_as(self.ps) for t in [0, 1]]))

    def analytic_mean(self):
        return self.ps

    def analytic_var(self):
        return self.ps * (1 - self.ps)
