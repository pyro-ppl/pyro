import torch
from pyro.distributions.distribution import Distribution


class TransformedDistribution(Distribution):
    """
    TransformedDistribution class
    """

    def __init__(self, base_distribution, bijector, *args, **kwargs):
        """
        Constructor; takes base distribution and bijector as arguments
        """
        super(TransformedDistribution, self).__init__(*args, **kwargs)
        # self.reparametrized = False #base_distribution.reparametrized
        self.reparametrized = base_distribution.reparametrized
        self.base_dist = base_distribution
        self.bijector = bijector

    def sample(self, *args, **kwargs):
        """
        sample from base and pass through bijector
        """
        x = self.base_dist.sample(*args, **kwargs)
        y = self.bijector(x)
        return y

    def log_pdf(self, y):
        x = self.bijector.inverse(y)
        log_pdf_1 = self.base_dist.log_pdf(x)
        log_pdf_2 = -self.bijector.log_det_jacobian(y)
        return log_pdf_1 + log_pdf_2

    def batch_log_pdf(self, y, batch_size=1):
        if y.dim() == 1 and batch_size == 1:
            return self.log_pdf(y)
        elif y.dim() == 1:
            y = y.expand(batch_size, y.size(0))
        x = self.bijector.inverse(y)
        log_pdf_1 = self.base_dist.batch_log_pdf(x)
        log_pdf_2 = -self.bijector.log_det_jacobian(y)
        return log_pdf_1 + log_pdf_2


class Bijector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Constructor for abstract class bijector
        """
        super(Bijector, self).__init__(*args, **kwargs)

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


class AffineExp(Bijector):
    def __init__(self, a_init, b_init, *args, **kwargs):
        """
        Constructor for univariate affine bijector followed by exp
        """
        super(AffineExp, self).__init__(*args, **kwargs)
        # if a_fixed:
        #    self.a = Variable(a_init)
        # else:
        #    self.a = Parameter(a_init)
        # if b_fixed and isinstance(b_init, torch.Tensor):
        #    self.b = Variable(b_init)
        # elif b_fixed and isinstance(b_init, Variable):
        #    self.b = b_init
        # else:
        #    self.b = Parameter(b_init)
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


"""
class AffineTanh(Bijector):
    def __init__(self, u_init, w_init, b_init, *args, **kwargs):
        #Constructor for affine tanh bijector
        super(AffineTanh, self).__init__(*args, **kwargs)
        self.u = Parameter(u_init)
        self.w = Parameter(w_init)
        self.b = Parameter(b_init)
        assert self.u.size()==self.w.size()

    def __call__(self, x, *args, **kwargs):
        y = x + torch.sum(self.u_hat() * torch.tanh(torch.sum(self.w*x) + self.b))
        return y

    def inverse(self, y, *args, **kwargs):
        #Virtual sample method.
        x_perp = y - x_par - self.u_hat() * torch.tanh(torch.sum(self.w*x_par) + self.b))

    def u_hat(self):
        def m(x):
            return torch.log(torch.ones(1)+torch.exp(x))-torch.ones(1) - x
        w_norm_sqr = torch.sum(torch.pow(self.w,2.0))
        w_dot_u = torch.sum(self.w*self.u)
        factor = m(w_dot_u) / w_norm_sqr
        return self.u + factor*self.w
"""
