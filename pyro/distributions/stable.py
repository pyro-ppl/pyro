import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution


def _standard_stable(shape, alpha, beta):
    # Implements a noisy-reparametrized sampler following the
    # Chambers-Mallows-Stuck method as corrected by Weron [1,3].
    with torch.no_grad():
        half_pi = math.pi / 2
        V = alpha.new_empty(shape).uniform_(-half_pi, half_pi)
        W = alpha.new_empty(shape).exponential_()
    inv_alpha = alpha.reciprocal()
    b = beta * alpha.mul(half_pi).tan()
    B = b.atan() * inv_alpha
    S = (1 + b * b).pow(0.5 * inv_alpha)
    v = alpha * (V + B)
    X = S * v.sin() / V.cos().pow(inv_alpha) \
        * ((V - v).cos() / W).pow((1-alpha) * inv_alpha)
    alpha_eq_1 = alpha == 1
    if alpha_eq_1.any():
        c = half_pi + beta * V
        X1 = 1/half_pi * (c * V.tan() - beta * (half_pi * W * V.cos() / c).log())
        # FIXME make this differentiable.
        X = torch.where(alpha_eq_1, X1, X)
    return X


class Stable(TorchDistribution):
    r"""
    Levy :math:`\alpha`-stable distribution. See [1] for a review.

    This uses Nolan's parametrization [2] of the ``loc`` parameter, which is
    required for differentiability. This corresponds to the notation
    :math:`S^0_\alpha(\beta,\sigma,\mu_0)` of [1], where :math:`\alpha` =
    stability, :math:`\beta` = skew, :math:`\sigma` = scale, and :math:`\mu_0`
    = loc.

    This implements a reparametrized sampler, but does not implement
    :meth:`log_prob` . Use in inference is thus limited to likelihood-free
    algorithms such as :class:`~pyro.infer.trace_crps.Trace_CRPS`.

    [1] S. Borak, W. Hardle, R. Weron (2005).
        Stable distributions.
        https://edoc.hu-berlin.de/bitstream/handle/18452/4526/8.pdf
    [2] J.P. Nolan (1997).
        Numerical calculation of stable densities and distribution functions.
    [3] Rafal Weron (1996).
        On the Chambers-Mallows-Stuck Method for
        Simulating Skewed Stable Random Variables.

    :param Tensor stability: Levy stability parameter :math:`\alpha\in(0,2]` .
    :param Tensor skew: Skewness :math:`\beta\in[-1,1]` .
    :param Tensor scale: Scale :math:`\sigma > 0` .
    :param Tensor loc: Location :math:`\mu_0` in Nolan's parametrization [2].
    """
    has_rsample = True
    arg_constraints = {"stability": constraints.interval(0, 2),  # half-open (0, 2]
                       "skew": constraints.interval(-1, 1),  # closed [-1, 1]
                       "scale": constraints.positive,
                       "loc": constraints.real}

    def __init__(self, stability, skew, scale, loc, validate_args=None):
        self.stability, self.skew, self.scale, self.loc = broadcast_all(
            stability, skew, scale, loc)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Stable, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        x = _standard_stable(shape, self.stability, self.skew)
        # Note this differs from [3] by using Nolan's parametrization [2].
        return self.loc + self.scale * (x - (2 / math.pi) * self.skew * self.scale.log())
