import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Delta(Distribution):
    """
    :param v: support element *(any)*

    Discrete distribution that assigns probability one to the single element in
    its support. Delta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.
    """
    enumerable = True

    def _sanitize_input(self, v):
        if v is not None:
            # stateless distribution
            return v
        elif self.v is not None:
            # stateful distribution
            return self.v
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, v=None, batch_size=1, *args, **kwargs):
        """
        Params:
          `v` - value
        """
        self.v = v
        if v is not None:
            if v.dim() == 1 and batch_size > 1:
                self.v = v.expand(v, v.size(0))
        super(Delta, self).__init__(*args, **kwargs)

    def sample(self, v=None, *args, **kwargs):
        v = self._sanitize_input(v)
        if isinstance(v, Variable):
            return v
        return Variable(v)

    def batch_log_pdf(self, x, v=None, batch_size=1, *args, **kwargs):
        v = self._sanitize_input(v)
        if x.dim == 1:
            x = x.expand(batch_size, x.size(0))
        # TODO: Add for AIR 'fudge_z_pres' mode -- clean up.
        ret = (torch.eq(x, v.expand_as(x)) - 1).float() * 999999
        return ret.view(-1)

    def log_pdf(self, x, v=None, *args, **kwargs):
        v = self._sanitize_input(v)
        if torch.equal(x.data, v.data.expand_as(x.data)):
            return Variable(torch.zeros(1).type_as(v.data))
        return Variable(torch.Tensor([-float("inf")]).type_as(v.data))

    def support(self, v=None, *args, **kwargs):
        """
        Returns the delta distribution's support, as a tensor along the first dimension.

        :param v: torch variable where each element of the tensor represents the point at
            which the delta distribution is concentrated.
        :return: torch variable enumerating the support of the delta distribution.
        :rtype: torch.autograd.Variable.
        """
        v = self._sanitize_input(v)
        # univariate case
        return Variable(v.data)
