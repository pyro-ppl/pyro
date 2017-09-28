import torch
import torch.nn as nn

from pyro.distributions.distribution import Distribution
from pyro.nn import AutoRegressiveNN
from pyro.util import ng_ones


class TransformedDistribution(Distribution):
    """
    :param base_distribution: distribution
    :param bijector: bijector

    Transforms the distribution with the bijector
    """

    def __init__(self, base_distribution, bijectors, *args, **kwargs):
        """
        Constructor; takes base distribution and bijector(s) as arguments
        """
        super(TransformedDistribution, self).__init__(*args, **kwargs)
        self.reparameterized = base_distribution.reparameterized
        self.base_dist = base_distribution
        if type(bijectors) is list or type(bijectors) is tuple:
           self.bijectors = bijectors
        else:
           self.bijectors = [bijectors]

    def sample(self, *args, **kwargs):
        """
        Sample from base and pass through bijector(s)
        """
        x = self.base_dist.sample(*args, **kwargs)
        next_input = x
        for bijector in self.bijectors:
           y = bijector(next_input)
           if bijector.add_inverse_to_cache:
               bijector.add_intermediate_to_cache(next_input, y, 'x')
           next_input = y
        return next_input

    def log_pdf(self, y, *args, **kwargs):
        """
        Scores the sample by inverting the bijector(s)
        """
        inverses = []
        next_to_invert = y
        for bijector in reversed(self.bijectors):
            inverse = bijector.inverse(next_to_invert)
            inverses.append(inverse)
            next_to_invert = inverse
        log_pdf_base = self.base_dist.log_pdf(inverses[-1], *args, **kwargs)
        log_det_jacobian = self.bijectors[-1].log_det_jacobian(y)
        for bijector, inverse in zip(list(reversed(self.bijectors))[1:], inverses[:-1]):
            log_det_jacobian += bijector.log_det_jacobian(inverse)
        return log_pdf_base - log_det_jacobian

class Bijector(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Constructor for abstract class bijector
        """
        super(Bijector, self).__init__(*args, **kwargs)
        self.add_inverse_to_cache = False

    def __call__(self, *args, **kwargs):
        # Virtual forward method
        raise NotImplementedError()

    def inverse(self, *args, **kwargs):
        # Virtual inverse method.
        raise NotImplementedError()

    def log_det_jacobian(self, *args, **kwargs):
        # Virtual logdet jacobian method.
        raise NotImplementedError()


class InverseAutoregressiveFlow(Bijector):
    """
    :param input_dim: NN input dimension
    :param hidden_dim: NN hidden dimension
    :param s_bias: bias default=`2.0`

    Inverse Autoregressive Flow
    """
    def __init__(self, input_dim, hidden_dim, s_bias=2.0):
        super(InverseAutoregressiveFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.arn = AutoRegressiveNN(input_dim, hidden_dim, output_dim_multiplier=2, output_bias=s_bias)
        self.sigmoid = nn.Sigmoid()
        self.intermediates_cache = {}
        self.add_inverse_to_cache = True

    def __call__(self, x, *args, **kwargs):
        """
        Invoke bijection x=>y
        """
        hidden = self.arn(x)
        sigma = self.sigmoid(hidden[:, 0:self.input_dim])
        mean = hidden[:, self.input_dim:]
        y = sigma * x + (ng_ones(sigma.size()) - sigma) * mean
        self.add_intermediate_to_cache(sigma, y, 'sigma')
        return y

    def inverse(self, y, *args, **kwargs):
        """
        Invert y => x
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
        """
        Calculates the determinant of the log jacobian
        """
        if (y, 'sigma') in self.intermediates_cache:
            sigma = self.intermediates_cache.pop((y, 'sigma'))
        else:
            raise KeyError("Bijector InverseAutoregressiveFlow expected to find" +
                           "key in intermediates cache but didn't")
        return torch.sum(torch.log(sigma))


class AffineExp(Bijector):
    """
    :param a_init: a
    :param b_init: b

    `y = exp(ax + b)`

    Univariate affine bijector followed by exp
    """
    def __init__(self, a_init, b_init):
        """
        Constructor for univariate affine bijector followed by exp
        """
        super(AffineExp, self).__init__()
        self.a = a_init
        self.b = b_init

    def __call__(self, x, *args, **kwargs):
        """
        Invoke bijection x=>y
        """
        y = self.a * x + self.b
        return torch.exp(y)

    def inverse(self, y, *args, **kwargs):
        """
        Invert y => x
        """
        x = (torch.log(y) - self.b) / self.a
        return x

    def log_det_jacobian(self, y, *args, **kwargs):
        """
        Calculates the determinant of the log jacobian
        """
        return torch.log(torch.abs(self.a)) + torch.log(y)
