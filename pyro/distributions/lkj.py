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
    Euclidean norm of each row is 1.
    """
    def check(self, value):
        unit_norm_row = (value.norm(dim=-1).sub(1) < 1e-6).min(-1)[0]
        return constraints.lower_cholesky.check(value) & unit_norm_row


# TODO rename this public interface to corr_cholesky if move upstream to pytorch
corr_cholesky_constraint = _CorrCholesky()


########################################
# Define transforms
########################################


class _PartialCorrToCorrLCholeskyTransform(Transform):
    """
    Transforms a vector of partial correlations into the cholesky factor of a
    correlation matrix.

    The input should have shape `[batch_shape] + [d * (d-1)/2]`. The output will have
    shape `[batch_shape + sample_shape] + [d, d]`.

    Reference:

    [1] `Cholesky Factors of Correlation Matrices`, Stan Reference Manual v2.18, Section 10.12
    """
    domain = constraints.interval(-1, 1)
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1
    event_shape = 1

    def __eq__(self, other):
        return isinstance(other, _PartialCorrToCorrLCholeskyTransform)

    def _call(self, z):
        D = (1.0 + math.sqrt(1.0 + 8.0 * z.shape[-1]))/2.0
        if D % 1 != 0:
            raise ValueError("Correlation matrix transformation requires d choose 2 inputs")
        D = int(D)

        x = torch.zeros(list(z.shape[:-1]) + [D,D], device=z.device)

        x[..., 0,0] = 1
        x[..., 1:,0] = z[..., :(D-1)]
        i = D - 1
        last_squared_x = torch.zeros(list(z.shape[:-1]) + [D], device=z.device)
        for j in range(1, D):
            distance_to_copy = D - 1 - j
            last_squared_x = last_squared_x[..., 1:] + x[...,j:,(j-1)].clone()**2
            x[..., j, j] = (1 - last_squared_x[..., 0]).sqrt()
            x[..., (j+1):, j] = z[..., i:(i + distance_to_copy)] * (1 - last_squared_x[..., 1:]).sqrt()
            i += distance_to_copy
        return x

    def _inverse(self, x):
        if (x.shape[-2] != x.shape[-1]):
            raise ValueError("A matrix that isn't square can't be a Cholesky factor of a correlation matrix")
        D = x.shape[-1]
        outlen = int(D * (D - 1) / 2)

        z_tri = torch.zeros(x.shape[:-2] + (D - 2, D - 2))
        z_stack = [
            x[..., 1:, 0]
        ]

        for i in range(2, D):
            z_tri[..., i - 2, 0:(i-1)] = x[..., i, 1:i] / (1-x[...,i,0:(i-1)].pow(2).cumsum(-1)).sqrt()
        for j in range(D - 2):
            z_stack.append(z_tri[..., j:, j])

        return torch.cat(z_stack, -1)

    def log_abs_det_jacobian(self, z, x):
        # This can probably be replaced with tril when support for
        # batched tril appears in pytorch 1.1
        # return (1 - x.tril(-1).pow(2).sum(-1)).log().sum(-1).mul(.5)
        mask = torch.eye(x.shape[-1], device=x.device).ne(1.0).to(dtype=x.dtype).expand_as(x)
        x_l = x * mask

        return (1 - x.pow(2).sum(-1)).log().sum(-1).mul(0.5)

class UnconstrainedToCorrLCholeskyTransform(Transform):
    domain = constraints.real
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1
    event_shape = 1
    _inner_transformation = _PartialCorrToCorrLCholeskyTransform()

    def __eq__(self, other):
        return isinstance(other, UnconstrainedToCorrLCholeskyTransform)

    def _call(self, y):
        z = y.tanh()
        return self._inner_transformation(z)

    def _inverse(self, x):
        z = self._inner_transformation._inverse(x)
        return torch.log((1 + z) / (1 - z)) / 2

    def log_abs_det_jacobian(self, y, x):
        return y.cosh().log().sum(-1).mul(-2) + self._inner_transformation.log_abs_det_jacobian(None, x)


# register transform to global store
@biject_to.register(corr_cholesky_constraint)
@transform_to.register(corr_cholesky_constraint)
def _transform_to_corr_cholesky(constraint):
    return UnconstrainedToCorrLCholeskyTransform()


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
    as this distribution, and \Omega is sampled from this distribution, scale_tril=diag(sqrt(\theta))*\Omega

    Note that the `event_shape` of this distribution is `[d, d]`

    :param int d: Dimensionality of the matrix
    :param torch.Tensor eta: A single positive number parameterizing the distribution.
    """
    arg_constraints = {"eta": constraints.positive}
    support = corr_cholesky_constraint
    has_rsample = False

    def __init__(self, d, eta):
        if not torch.is_tensor(eta):
            eta = torch.FloatTensor([eta])
        if any(eta <= 0):
            raise ValueException("eta must be > 0")
        vector_size = (d * (d - 1)) // 2
        alpha = eta.add(0.5 * (d  - 1.0))

        concentrations = eta.new().resize_(vector_size)
        i = 0
        for k in range(d-1):
            alpha -= .5
            for j in range(k+1, d):
                concentrations[i] = alpha
                i += 1
        self._generating_distribution = Beta(concentrations, concentrations)
        self._transformation = _PartialCorrToCorrLCholeskyTransform()
        self._eta = eta
        self._d = d
        self._lkj_constant = None
        self._event_shape = torch.Size((d, d))

    def sample(self, *args, **kwargs):
        return self._transformation(self._generating_distribution.sample(*args, **kwargs).detach().mul(2).add(-1.0))

    def lkj_constant(self, eta, K):
        if self._lkj_constant is not None:
            return self._lkj_constant

        Km1 = K - 1

        constant = torch.lgamma(eta.add(0.5 * Km1)).mul(Km1)

        k = torch.linspace(start=1, end=Km1, steps=Km1, device=eta.device)
        constant -= (k.mul(math.log(math.pi) * 0.5) + torch.lgamma(eta.add( 0.5 * (Km1 - k)))).sum()

        self._lkj_constant = constant
        return constant

    def log_prob(self, x):
        eta = self._eta

        lp = self.lkj_constant(eta, self._d)

        Km1 = x.shape[-1] - 1
        log_diagonals = x.diagonal(offset=0, dim1=-1, dim2=-2)[..., :-1].log()
        #TODO: Confirm that this should be a 0-indexed rather than 1-indexed vector
        values = log_diagonals * torch.linspace(start=Km1 - 1, end=0, steps=Km1, device=x.device).expand_as(log_diagonals)

        values += log_diagonals.mul(eta.mul(2).add(-2.0))
        return values.sum(-1) + lp
