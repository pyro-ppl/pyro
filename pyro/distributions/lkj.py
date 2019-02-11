from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.distributions.constraints import Constraint
from torch.distributions.transforms import Transform, ComposeTransform

from pyro.distributions import Beta, TorchDistribution


########################################
# Define constraint
########################################


class _CorrCholesky(Constraint):
    """
    Constrains to lower-triangular square matrices with positive diagonals and
    Euclidean norm of each row is 1, such that `torch.mm(omega, omega.t())` will
    have unit diagonal.
    """
    def check(self, value):
        unit_norm_row = (value.norm(dim=-1).sub(1) < 1e-6).min(-1)[0]
        return constraints.lower_cholesky.check(value) & unit_norm_row


# TODO rename this public interface to corr_cholesky if move upstream to pytorch
corr_cholesky_constraint = _CorrCholesky()


########################################
# Define transforms
########################################

def _vector_to_l_cholesky(z):
    D = (1.0 + math.sqrt(1.0 + 8.0 * z.shape[-1]))/2.0
    if D % 1 != 0:
        raise ValueError("Correlation matrix transformation requires d choose 2 inputs")
    D = int(D)
    x = z.new_zeros(list(z.shape[:-1]) + [D,D])

    x[..., 0,0] = 1
    x[..., 1:,0] = z[..., :(D-1)]
    i = D - 1
    last_squared_x = z.new_zeros(list(z.shape[:-1]) + [D])
    for j in range(1, D):
        distance_to_copy = D - 1 - j
        last_squared_x = last_squared_x[..., 1:] + x[...,j:,(j-1)].clone()**2
        x[..., j, j] = (1 - last_squared_x[..., 0]).sqrt()
        x[..., (j+1):, j] = z[..., i:(i + distance_to_copy)] * (1 - last_squared_x[..., 1:]).sqrt()
        i += distance_to_copy
    return x

# Note on the bijectivity of CorrLCholeskyTransform:
# The transform has two phases - first, from the domain of reals to (-1, 1)
# and second, from a vector in (-1, 1) to a lower cholesky factorself.
# The second part of the transform cannot operate correctly if the inputs
# consist of many 1s and -1s. This leads to a numerical precision issue with the
# tanh transformation, where for inputs far from 0 tanh produces a -1 or 1.
# For that reason, the output of the tanh operation is clamped, and the
# transformation is not perfectly bijective.

class CorrLCholeskyTransform(Transform):
    """
    Transforms a vector into the cholesky factor of a correlation matrix.

    The input should have shape `[batch_shape] + [d * (d-1)/2]`. The output will have
    shape `[batch_shape] + [d, d]`.

    Reference:

    [1] `Cholesky Factors of Correlation Matrices`, Stan Reference Manual v2.18, Section 10.12
    """
    domain = constraints.real
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1
    event_dim = 1

    def __eq__(self, other):
        return isinstance(other, CorrLCholeskyTransform)

    def _call(self, x):
        # Note: Limiting eps to finfo.eps was insufficient to resolve the issue
        # described above
        #eps = torch.finfo(x.dtype).eps
        eps = 1e-4
        z = x.tanh().clamp(-1 + eps, 1 - eps)
        return _vector_to_l_cholesky(z)

    def _inverse(self, y):
        if (y.shape[-2] != y.shape[-1]):
            raise ValueError("A matrix that isn't square can't be a Cholesky factor of a correlation matrix")
        D = y.shape[-1]

        z_tri = y.new_zeros(y.shape[:-2] + (D - 2, D - 2))
        z_stack = [
            y[..., 1:, 0]
        ]

        for i in range(2, D):
            z_tri[..., i - 2, 0:(i-1)] = y[..., i, 1:i] / (1-y[...,i,0:(i-1)].pow(2).cumsum(-1)).sqrt()
        for j in range(D - 2):
            z_stack.append(z_tri[..., j:, j])

        z = torch.cat(z_stack, -1)
        return torch.log1p((2*z)/(1-z)) / 2

    def log_abs_det_jacobian(self, x, y):
        # Note dependence on pytorch 1.0.1 for batched tril
        tanpart = x.cosh().log().sum(-1).mul(-2)
        matpart = (1 - y.pow(2).cumsum(-1).tril(diagonal=-2)).log().div(2).sum(-1).sum(-1)
        return tanpart + matpart

# register transform to global store
@biject_to.register(corr_cholesky_constraint)
@transform_to.register(corr_cholesky_constraint)
def _transform_to_corr_cholesky(constraint):
    return CorrLCholeskyTransform()


########################################
# Define distribution
########################################

# TODO: Modify class to support more than one eta value at a time?
class CorrLCholeskyLKJPrior(TorchDistribution):
    """
    Generates cholesky factors of correlation matrices using an LKJ prior.

    The expected use is to combine it with a vector of variances and pass it
    to the scale_tril parameter of a multivariate distribution such as MultivariateNormal.

    E.g., if \theta is a (positive) vector of covariances with the same dimensionality
    as this distribution, and \Omega is sampled from this distribution, scale_tril=torch.mm(torch.diag(sqrt(\theta)), \Omega)

    Note that the `event_shape` of this distribution is `[d, d]`

    :param int d: Dimensionality of the matrix
    :param torch.Tensor eta: A single positive number parameterizing the distribution.
    """
    arg_constraints = {"eta": constraints.positive}
    support = corr_cholesky_constraint
    has_rsample = False

    def __init__(self, d, eta):
        if not torch.is_tensor(eta):
            raise ValueError("Eta must be a tensor")
        if any(eta <= 0):
            raise ValueError("eta must be > 0")
        vector_size = (d * (d - 1)) // 2
        alpha = eta.add(0.5 * (d  - 1.0))

        concentrations = eta.new_empty(vector_size,)
        i = 0
        for k in range(d-1):
            alpha -= .5
            for j in range(k+1, d):
                concentrations[..., i] = alpha
                i += 1
        self._generating_distribution = Beta(concentrations, concentrations)
        self._eta = eta
        self._d = d
        self._lkj_constant = None
        self._event_shape = torch.Size((d, d))

    def sample(self, *args, **kwargs):
        z = self._generating_distribution.sample(*args, **kwargs).detach().to(dtype=self._eta.dtype).mul(2).add(-1.0)
        return _vector_to_l_cholesky(z)

    def lkj_constant(self, eta, K):
        if self._lkj_constant is not None:
            return self._lkj_constant

        Km1 = K - 1

        constant = torch.lgamma(eta.add(0.5 * Km1)).mul(Km1)

        k = torch.linspace(start=1, end=Km1, steps=Km1, dtype=eta.dtype, device=eta.device)
        constant -= (k.mul(math.log(math.pi) * 0.5) + torch.lgamma(eta.add( 0.5 * (Km1 - k)))).sum()

        self._lkj_constant = constant
        return constant

    def log_prob(self, x):
        eta = self._eta

        lp = self.lkj_constant(eta, self._d)

        Km1 = x.shape[-1] - 1
        log_diagonals = x.diagonal(offset=0, dim1=-1, dim2=-2)[..., 1:].log()
        values = log_diagonals * torch.linspace(start=Km1 - 1, end=0, steps=Km1, dtype=x.dtype, device=x.device).expand_as(log_diagonals)

        values += log_diagonals.mul(eta.mul(2).add(-2.0))
        return values.sum(-1) + lp
