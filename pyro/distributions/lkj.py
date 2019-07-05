from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.distributions.constraints import Constraint
from torch.distributions.transforms import Transform

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
        unit_norm_row = (value.norm(dim=-1).sub(1) < 1e-4).min(-1)[0]
        return constraints.lower_cholesky.check(value) & unit_norm_row


corr_cholesky_constraint = _CorrCholesky()


########################################
# Define transforms
########################################

def _vector_to_l_cholesky(z):
    D = (1.0 + math.sqrt(1.0 + 8.0 * z.shape[-1])) / 2.0
    if D % 1 != 0:
        raise ValueError("Correlation matrix transformation requires d choose 2 inputs")
    D = int(D)
    x = torch.zeros(z.shape[:-1] + (D, D), dtype=z.dtype, device=z.device)

    x[..., 0, 0] = 1
    x[..., 1:, 0] = z[..., :(D - 1)]
    i = D - 1
    last_squared_x = torch.zeros(z.shape[:-1] + (D,), dtype=z.dtype, device=z.device)
    for j in range(1, D):
        distance_to_copy = D - 1 - j
        last_squared_x = last_squared_x[..., 1:] + x[..., j:, (j - 1)].clone()**2
        x[..., j, j] = (1 - last_squared_x[..., 0]).sqrt()
        x[..., (j + 1):, j] = z[..., i:(i + distance_to_copy)] * (1 - last_squared_x[..., 1:]).sqrt()
        i += distance_to_copy
    return x


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
        z = x.tanh()
        return _vector_to_l_cholesky(z)

    def _inverse(self, y):
        if (y.shape[-2] != y.shape[-1]):
            raise ValueError("A matrix that isn't square can't be a Cholesky factor of a correlation matrix")
        D = y.shape[-1]

        z_tri = torch.zeros(y.shape[:-2] + (D - 2, D - 2), dtype=y.dtype, device=y.device)
        z_stack = [
            y[..., 1:, 0]
        ]

        for i in range(2, D):
            z_tri[..., i - 2, 0:(i - 1)] = y[..., i, 1:i] / (1 - y[..., i, 0:(i - 1)].pow(2).cumsum(-1)).sqrt()
        for j in range(D - 2):
            z_stack.append(z_tri[..., j:, j])

        z = torch.cat(z_stack, -1)
        return torch.log1p((2 * z) / (1 - z)) / 2

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
class LKJCorrCholesky(TorchDistribution):
    """
    Generates cholesky factors of correlation matrices using an LKJ prior.

    The expected use is to combine it with a vector of variances and pass it
    to the scale_tril parameter of a multivariate distribution such as MultivariateNormal.

    E.g., if theta is a (positive) vector of covariances with the same dimensionality
    as this distribution, and Omega is sampled from this distribution,
    scale_tril=torch.mm(torch.diag(sqrt(theta)), Omega)

    Note that the `event_shape` of this distribution is `[d, d]`

    .. note::

       When using this distribution with HMC/NUTS, it is important to
       use a `step_size` such as 1e-4. If not, you are likely to experience LAPACK
       errors regarding positive-definiteness.

    For example usage, refer to
    `pyro/examples/lkj.py <https://github.com/pyro-ppl/pyro/blob/dev/examples/lkj.py>`_.

    :param int d: Dimensionality of the matrix
    :param torch.Tensor eta: A single positive number parameterizing the distribution.
    """
    arg_constraints = {"eta": constraints.positive}
    support = corr_cholesky_constraint
    has_rsample = False

    def __init__(self, d, eta, validate_args=None):
        if eta.numel() != 1:
            raise ValueError("eta must be a single number; for a larger batch size, call expand")
        if d <= 1:
            raise ValueError("d must be > 1 in any correlation matrix")
        eta = eta.squeeze()
        vector_size = (d * (d - 1)) // 2
        alpha = eta.add(0.5 * (d - 1.0))

        concentrations = torch.empty(vector_size, dtype=eta.dtype, device=eta.device)
        i = 0
        for k in range(d - 1):
            alpha -= .5
            concentrations[..., i:(i + d - k - 1)] = alpha
            i += d - k - 1
        self._gen = Beta(concentrations, concentrations)
        self.eta = eta
        self._d = d
        self._lkj_constant = None
        super(LKJCorrCholesky, self).__init__(torch.Size(), torch.Size((d, d)), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        y = self._gen.sample(sample_shape=self.batch_shape + sample_shape).detach()
        z = y.mul(2).add(-1.0)
        return _vector_to_l_cholesky(z)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCorrCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new._gen = self._gen
        new.eta = self.eta
        new._d = self._d
        new._lkj_constant = self._lkj_constant
        super(LKJCorrCholesky, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def lkj_constant(self, eta, K):
        if self._lkj_constant is not None:
            return self._lkj_constant

        Km1 = K - 1

        constant = torch.lgamma(eta.add(0.5 * Km1)).mul(Km1)

        k = torch.linspace(start=1, end=Km1, steps=Km1, dtype=eta.dtype, device=eta.device)
        constant -= (k.mul(math.log(math.pi) * 0.5) + torch.lgamma(eta.add(0.5 * (Km1 - k)))).sum()

        self._lkj_constant = constant
        return constant

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)

        eta = self.eta

        lp = self.lkj_constant(eta, self._d)

        Km1 = self._d - 1

        log_diagonals = x.diagonal(offset=0, dim1=-1, dim2=-2)[..., 1:].log()
        # TODO: Figure out why the `device` kwarg to torch.linspace seems to not work in certain situations,
        # and a seemingly redundant .to(x.device) is needed below.
        values = log_diagonals * torch.linspace(start=Km1 - 1, end=0, steps=Km1,
                                                dtype=x.dtype,
                                                device=x.device).expand_as(log_diagonals).to(x.device)

        values += log_diagonals.mul(eta.mul(2).add(-2.0))
        values = values.sum(-1) + lp
        values, _ = torch.broadcast_tensors(values, torch.empty(self.batch_shape))
        return values
