from __future__ import absolute_import, division, print_function

import math

from torch.distributions import constraints
from torch.distributions.constraints import Constraint
from torch.distributions.transforms import Transform

from pyro.distributions import Beta, TorchDistribution


class _CorrCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals and Euclidean
    norm of each row is 1.
    """
    def check(self, value):
        value_tril = value.tril()
        lower_triangular = (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]

        positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]

        unit_norm_row = ((value.pow(2).sum(-1) - 1).abs() < 1e-6).min(-1)[0]
        return lower_triangular & positive_diagonal & unit_norm_row


# TODO rename this to corr_cholesky if move upstream to pytorch
corr_cholesky_constraint = _CorrCholesky()


class CorrCholeskyTransform(Transform):
    """
    Transforms a vector of canonical partial correlations into the cholesky factor of
    a covariance matrix.

    Note that this transformation assumes that the vector of cpc's is already on the interval
    [-1, 1].
    """
    domain = constraints.interval(-1, 1)
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, CorrCholeskyTransform)

    def _call(self, z):
        D = (1.0 + math.sqrt(1.0 + 8.0 * z.shape[0]))/2.0
        if D % 1 != 0:
            raise ValueError("Correlation matrix transformation requires d choose 2 inputs")
        D = int(D)

        x = torch.zeros((D,D), device=z.device)

        x[0,0] = 1
        x[1:,0] = current_x = z[:(D-1)]
        i = D - 1
        last_squared_x = None
        for j in range(1, D):
            distance_to_copy = D - 1 - j
            new_z = z[i:(i + distance_to_copy)]
            if last_squared_x is None:
                last_squared_x = current_x**2
            else:
                last_squared_x = last_squared_x[1:] + current_x**2
            x[j, j] = (1 - last_squared_x[0]).sqrt()
            current_x = new_z * (1 - last_squared_x[1:]).sqrt()
            x[(j+1):, j] = current_x
            i += distance_to_copy

        return x

    def _inverse(self, x):
        if (x.shape[0] != x.shape[1]):
            raise ValueError("A matrix that isn't square can't be a Cholesky factor of a correlation matrix")
        D = x.shape[0]

        z_stack = [
            x[1:, 0]
        ]
        current_x = z_stack[0]
        last_squared_x = None
        for j in range(1, D):
            if last_squared_x is None:
                last_squared_x = current_x**2
            else:
                last_squared_x += current_x[1:]**2
        current_x = x[j:, j]
        z_stack.append(current_x / (1 - last_squared_x).sqrt())
        z = torch.cat(z_stack)
        return z

    def log_abs_det_jacobian(self, x, z):
        return (1 - x.tril(-1).pow(2).sum(1)).log().sum() * .5


class UnconstrainedLowerCholeskyCorrTransform(CorrCholeskyTransform):
    """
    Transforms a vector of reals into the cholesky factor of
    a covariance matrix.

    Note that this transformation does not assume that the vector of cpc's is already on the interval
    [-1, 1].
    """
    domain = constraints.real
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, UnconstrainedLowerCholeskyCorrTransform)

    def _call(self, y):
        return super(UnconstrainedLowerCholeskyCorrTransform, self)(y.tanh())

    def _inverse(self, x):
        z = super(UnconstrainedLowerCholeskyCorrTransform, self)(x)
        return torch.log((z + 1) / (1 - z))/2

    def log_abs_det_jacobian(self, x, y):
        transformation_part = super(UnconstrainedLowerCholeskyCorrTransform, self)(x, y.tanh())
        tanh_jacobian = y.cosh().log().sum() * -2
        log_abs_det = transformation_part + tanh_jacobian
        return log_abs_det


class LKJCholeskyFactor(TorchDistribution):
    """
    Generates cholesky factors of correlation matrices using an LKJ prior.

    The expected use is to combine it with a vector of variances and pass it
    to the scale_tril parameter of a multivariate distribution such as MultivariateNormal.

    E.g., if \theta is a (positive) vector of covariances with the same dimensionality
    as this distribution, and \Omega is sampled from this distribution, scale_tril=diag(sqrt(\theta))*\Omega

    :param int d: Dimensionality of the matrix
    :param torch.Tensor eta: A single positive number parameterizing the distribution.
    """
    def __init__(self, d, eta):
        if not torch.is_tensor(eta):
            eta = torch.FloatTensor([eta])
        if any(eta <= 0):
            raise ValueException("eta must be > 0")
        vector_size = (d * (d - 1)) // 2
        alpha = eta.add(0.5 * (d  - 1.0))

        concentrations = eta.new().expand(vector_size)
        i = 0
        for k in range(d-1):
            alpha -= .5
            for j in range(k+1, d):
                concentrations[i] = alpha
                i += 1
        self._generating_distribution = Beta(concentrations, concentrations)
        self._transformation = CorrCholeskyTransform()
        self._eta = eta
        self._d = d
        self._lkj_constant = None

    def sample(self):
        return self._transformation(self._generating_distribution.sample().mul(2).add(- 1.0))

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

        lp = lkj_constant(eta, x.shape[1])

        Km1 = x.shape[1] - 1
        log_diagonals = x.diag().tail(Km1).log()
        values = torch.linspace(start=Km1 - 1, end=0, steps=km1, device=x.device) * log_diagonals

        values += log_diagonals.mul(eta.mul(2).add(-2.0))

        return values.sum() + lp
