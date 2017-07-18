import numpy as np
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class Multinomial(Distribution):
    """
    Multinomial distributino
    """

    def __init__(self, ps, n, batch_size=1, *args, **kwargs):
        """
        Params:
          ps - probabilities
          n - num trials
        """
        if ps.dim() == 1 and batch_size > 1:
            self.ps = ps.unsqueeze(0).expand(batch_size, ps.size(0))
            self.n = n.unsqueeze(0).expand(batch_size, n.size(0))
        else:
            self.ps = ps
            self.n = n
        super(Multinomial, self).__init__(*args, **kwargs)

    def sample(self):
        _, counts = np.unique([self.expanded_sample().data.numpy()], return_counts=True)
        return Variable(torch.Tensor(counts))

    def expanded_sample(self):
        # get the int from Variable or Tensor
        if self.n.data.dim() == 2:
            _n = int(self.n.data[0][0])
        else:
            _n = int(self.n.data[0])
        return Variable(torch.multinomial(self.ps.data, _n, replacement=True))

    def batch_log_pdf(self, x, batch_size=1):
        """
        hack replacement for batching multinomail score
        """
        # FIXME: torch.split so tensor is differentiable
        if x.dim() == 1:
            x = x.expand(batch_size, 0)
        out_arr = [[self._get_tensor(self.log_pdf([
                    x.narrow(0, ix, ix + 1),
                    self.ps.narrow(0, ix, ix + 1)
                    ]))[0]]
                   for ix in range(int(x.size(0)))]
        return Variable(torch.Tensor(out_arr))

    def log_pdf(self, _x):
        """
        Multinomial log-likelihood
        """
        # probably use gamma function when added
        if isinstance(_x, list):
            x = _x[0]
            ps = _x[1]
        else:
            x = _x
            ps = self.ps
        prob = torch.sum(torch.mul(x, torch.log(ps)))
        logfactsum = self._log_factorial(torch.sum(x))
        # this is disgusting because theres no clean way to do this ..yet
        logfactct = torch.sum(Variable(torch.Tensor(
            [self._log_factorial(_xi) for _xi in self._get_array(x)])))
        return prob + logfactsum - logfactct

#     https://stackoverflow.com/questions/13903922/multinomial-pmf-in-python-scipy-numpy
    def _log_factorial(self, var_s):
        if isinstance(var_s, Variable):
            var_s = int(var_s.data[0])
        if isinstance(var_s, torch.Tensor):
            var_s = int(var_s[0])
        else:
            var_s = int(var_s)
        xs = Variable(torch.Tensor(range(1, var_s + 1)))
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
