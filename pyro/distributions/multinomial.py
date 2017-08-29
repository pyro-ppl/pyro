import numpy as np
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class Multinomial(Distribution):
    """
    :param ps: probabilities *(real array with elements that sum to one)*
    :param n: number of trials *(int (>=1))*

    Distribution over counts for ``n`` independent ``Discrete({ps: ps})``
    trials.
    """
    def _sanitize_input(self, ps, n):
        if ps is not None:
            # stateless distribution
            return ps, n
        elif self.ps is not None:
            # stateful distribution
            return self.ps, self.n
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, ps=None, n=None, batch_size=1, *args, **kwargs):
        """
        Params:
          ps - probabilities
          n - num trials
        """
        self.ps = ps
        self.n = n
        if ps is not None:
            if ps.dim() == 1 and batch_size > 1:
                self.ps = ps.expand(batch_size, ps.size(0))
                self.n = n.expand(batch_size, n.size(0))
        super(Multinomial, self).__init__(*args, **kwargs)

    def sample(self, ps=None, n=None, *args, **kwargs):
        _ps, _n = self._sanitize_input(ps, n)
        _, counts = np.unique([self.expanded_sample().data.numpy()], return_counts=True)
        return Variable(torch.Tensor(counts).type_as(_ps.data))

    def expanded_sample(self, ps=None, n=None, *args, **kwargs):
        _ps, _n = self._sanitize_input(ps, n)
        # get the int from Variable or Tensor
        if _n.data.dim() == 2:
            _n = int(_n.data[0][0])
        else:
            _n = int(_n.data[0])
        return Variable(torch.multinomial(_ps.data, _n, replacement=True).type_as(_ps.data))

    def batch_log_pdf(self, x, ps=None, n=None, batch_size=1, *args, **kwargs):
        """
        hack replacement for batching multinomail score
        """
        # FIXME: torch.split so tensor is differentiable
        _ps, _n = self._sanitize_input(ps, n)
        if x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        out_arr = [[self._get_tensor(self.log_pdf([
                    x.narrow(0, ix, ix + 1),
                    _ps.narrow(0, ix, ix + 1)
                    ], _ps, _n))[0]]
                   for ix in range(int(x.size(0)))]
        return Variable(torch.Tensor(out_arr).type_as(_ps.data))

    def log_pdf(self, _x, ps=None, n=None, *args, **kwargs):
        """
        Multinomial log-likelihood
        """
        # probably use gamma function when added
        _ps, _n = self._sanitize_input(ps, n)
        ttype = _ps.data.type()
        if isinstance(_x, list):
            x = _x[0]
            ps = _x[1]
        else:
            x = _x
            ps = _ps
        prob = torch.sum(torch.mul(x, torch.log(ps)))
        logfactsum = self._log_factorial(torch.sum(x), ttype)
        # this is disgusting because theres no clean way to do this ..yet
        logfactct = torch.sum(Variable(torch.Tensor(
            [self._log_factorial(_xi, ttype) for _xi in self._get_array(x)])
            .type_as(_ps.data)))
        return prob + logfactsum - logfactct

#     https://stackoverflow.com/questions/13903922/multinomial-pmf-in-python-scipy-numpy
    def _log_factorial(self, var_s, tensor_type):
        if isinstance(var_s, Variable):
            var_s = int(var_s.data[0])
        if isinstance(var_s, torch.Tensor):
            var_s = int(var_s[0])
        else:
            var_s = int(var_s)
        xs = Variable(torch.Tensor(range(1, var_s + 1)).type(tensor_type))
        return torch.sum(torch.log(xs)).data[0]

    def _get_tensor(self, var):
        if isinstance(var, Variable):
            return var.data
        return var

    def _get_array(self, var):
        if var.data.dim() == 1:
            return var.data
        # nested tensor arrays because of batches"
        return var.data.numpy()[0]
