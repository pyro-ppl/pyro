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

    def _sanitize_input(self, ps):
        if ps is not None:
            # stateless distribution
            return ps
        elif self.ps is not None:
            # stateful distribution
            return self.ps
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, ps=None, batch_size=1, *args, **kwargs):
        """
        Params:
          ps = tensor of probabilities
        """
        self.ps = ps
        if ps is not None:
            if ps.dim() == 1:
                self.ps = ps.expand(batch_size, ps.size(0))
        super(Bernoulli, self).__init__(*args, **kwargs)

    def sample(self, ps=None, *args, **kwargs):
        """
        Bernoulli sampler.
        """
        ps = self._sanitize_input(ps)
        return Variable(torch.bernoulli(ps.data).type_as(ps.data))

    def batch_log_pdf(self, x, ps=None, batch_size=1, *args, **kwargs):
        ps = self._sanitize_input(ps)
        if x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        if ps.size() != x.size():
            ps = ps.expand_as(x)
        x_1 = x - 1
        ps_1 = ps - 1
        xmul = torch.mul(x, ps)
        xmul_1 = torch.mul(x_1, ps_1)
        logsum = torch.log(torch.add(xmul, xmul_1))
        return torch.sum(logsum, 1)

    def support(self, ps=None, *args, **kwargs):
        """
        Returns the Bernoulli distribution's support, as a tensor along the first dimension.

        Note that this returns support values of all the batched RVs in lock-step, rather
        than the full cartesian product. To iterate over the cartesian product, you must
        construct univariate Bernoullis and use itertools.product() over all univariate
        variables (may be expensive).

        :param ps: torch variable where each element of the tensor denotes the probability of
            and independent event.
        :return: torch variable enumerating the support of the Bernoulli distribution.
            Each item in the return value, when enumerated along the first dimensions, yields a
            value from the distribution's support which has the same dimension as would be returned by
            sample.
        :rtype: torch.autograd.Variable.
        """
        ps = self._sanitize_input(ps)
        return Variable(torch.stack([torch.Tensor([t]).expand_as(ps) for t in [0, 1]]))

    def analytic_mean(self, ps=None):
        ps = self._sanitize_input(ps)
        return ps

    def analytic_var(self, ps=None):
        ps = self._sanitize_input(ps)
        return ps * (1 - ps)
