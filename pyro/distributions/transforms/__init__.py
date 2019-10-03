from torch.distributions import biject_to, transform_to
from torch.distributions.transforms import *  # noqa F403
from torch.distributions.transforms import __all__ as torch_transforms

from pyro.distributions.constraints import IndependentConstraint, corr_cholesky_constraint
from pyro.distributions.transforms.affine_coupling import AffineCoupling
from pyro.distributions.transforms.batch_norm import BatchNormTransform
from pyro.distributions.transforms.block_autoregressive import BlockAutoregressive
from pyro.distributions.transforms.cholesky import CorrLCholeskyTransform
from pyro.distributions.transforms.householder import HouseholderFlow
from pyro.distributions.transforms.iaf import InverseAutoregressiveFlow, InverseAutoregressiveFlowStable
from pyro.distributions.transforms.neural_autoregressive import (ELUTransform, LeakyReLUTransform, NeuralAutoregressive,
                                                                 TanhTransform)
from pyro.distributions.transforms.permute import PermuteTransform
from pyro.distributions.transforms.planar import ConditionalPlanarFlow, PlanarFlow
from pyro.distributions.transforms.polynomial import PolynomialFlow
from pyro.distributions.transforms.radial import RadialFlow
from pyro.distributions.transforms.sylvester import SylvesterFlow

########################################
# register transforms

biject_to.register(IndependentConstraint, lambda c: biject_to(c.base_constraint))
transform_to.register(IndependentConstraint, lambda c: transform_to(c.base_constraint))


@biject_to.register(corr_cholesky_constraint)
@transform_to.register(corr_cholesky_constraint)
def _transform_to_corr_cholesky(constraint):
    return CorrLCholeskyTransform()


__all__ = [
    'AffineCoupling',
    'BatchNormTransform',
    'BlockAutoregressive',
    'ConditionalPlanarFlow',
    'CorrLCholeskyTransform',
    'ELUTransform',
    'HouseholderFlow',
    'InverseAutoregressiveFlow',
    'InverseAutoregressiveFlowStable',
    'LeakyReLUTransform',
    'NeuralAutoregressive',
    'PermuteTransform',
    'PlanarFlow',
    'PolynomialFlow',
    'RadialFlow',
    'SylvesterFlow',
    'TanhTransform',
]

__all__.extend(torch_transforms)
del torch_transforms
