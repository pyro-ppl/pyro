import torch
import torch.nn as nn
from pyro.distributions.distribution import Distribution
from pyro.nn import MaskedLinear, AutoRegressiveNN
from torch.autograd import Variable
from pyro.util import ng_ones, ng_zeros


class TransformedDistribution(Distribution):
    """
    TransformedDistribution class
    """

    def __init__(self, base_distribution, bijector, *args, **kwargs):
        """
        Constructor; takes base distribution and bijector as arguments
        """
        super(TransformedDistribution, self).__init__(*args, **kwargs)
        self.reparameterized = base_distribution.reparameterized
        self.base_dist = base_distribution
        self.bijector = bijector

    def sample(self, *args, **kwargs):
        """
        sample from base and pass through bijector
        """
        x = self.base_dist.sample(*args, **kwargs)
        y = self.bijector(x)
        if self.bijector.add_inverse_to_cache:
            self.bijector.add_intermediate_to_cache(x, y, 'x')
        return y

    def log_pdf(self, y, *args, **kwargs):
        x = self.bijector.inverse(y)
        log_pdf_1 = self.base_dist.log_pdf(x, *args, **kwargs)
        log_pdf_2 = -self.bijector.log_det_jacobian(y)
        return log_pdf_1 + log_pdf_2


class Bijector(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Constructor for abstract class bijector
        """
        super(Bijector, self).__init__(*args, **kwargs)
        self.add_inverse_to_cache = False

    def __call__(self, *args, **kwargs):
        """
        Virtual forward method
        """
        raise NotImplementedError()

    def inverse(self, *args, **kwargs):
        """
        Virtual inverse method.
        """
        raise NotImplementedError()

    def log_det_jacobian(self, *args, **kwargs):
        """
        Virtual logdet jacobian method.
        """
        raise NotImplementedError()


class InverseAutoregressiveFlow(Bijector):
    def __init__(self, input_dim, hidden_dim, s_bias=2.0):
        super(InverseAutoregressiveFlow, self).__init__()
        self.arn_s = AutoRegressiveNN(input_dim, hidden_dim, output_bias=s_bias)
        self.arn_m = AutoRegressiveNN(input_dim, hidden_dim,
                                      lin1=self.arn_s.get_lin1(),
                                      mask_encoding=self.arn_s.get_mask_encoding())
        self.sigmoid = nn.Sigmoid()
        self.intermediates_cache = {}
        self.add_inverse_to_cache = True

    def __call__(self, x, *args, **kwargs):
        """
        invoke bijection x=>y
        """
        s = self.arn_s(x)
        sigma = self.sigmoid(s)
        m = self.arn_m(x)
        y = sigma * x + (ng_ones(sigma.size()) - sigma) * m
        self.add_intermediate_to_cache(sigma, y, 'sigma')
        return y

    def inverse(self, y, *args, **kwargs):
        """
        invert y => x
        """
        if (y, 'x') in self.intermediates_cache:
            x = self.intermediates_cache.pop((y, 'x'))
            return x
        else:
            raise KeyError("Bijector InverseAutoregressiveFlow expected to find" +
                           "key in intermediates cache but didn't")

    def add_intermediate_to_cache(self, intermediate, y, name):
        assert((y, name) not in self.intermediates_cache),\
            "key collision in add_intermediate_to_cache"
        self.intermediates_cache[(y, name)] = intermediate

    def log_det_jacobian(self, y, *args, **kwargs):
        if (y, 'sigma') in self.intermediates_cache:
            sigma = self.intermediates_cache.pop((y, 'sigma'))
        else:
            raise KeyError("Bijector InverseAutoregressiveFlow expected to find" +
                           "key in intermediates cache but didn't")
        return torch.sum(torch.log(sigma))


class AffineExp(Bijector):
    def __init__(self, a_init, b_init):
        """
        Constructor for univariate affine bijector followed by exp
        """
        super(AffineExp, self).__init__()
        self.a = a_init
        self.b = b_init

    def __call__(self, x, *args, **kwargs):
        """
        invoke bijection x=>y
        """
        y = self.a * x + self.b
        return torch.exp(y)

    def inverse(self, y, *args, **kwargs):
        """
        invert y => x
        """
        x = (torch.log(y) - self.b) / self.a
        return x

    def log_det_jacobian(self, y, *args, **kwargs):
        return torch.log(torch.abs(self.a)) + torch.log(y)
