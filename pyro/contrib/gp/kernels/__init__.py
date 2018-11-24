from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.kernels.brownian import Brownian
from pyro.contrib.gp.kernels.coregionalize import Coregionalize
from pyro.contrib.gp.kernels.dot_product import DotProduct, Linear, Polynomial
from pyro.contrib.gp.kernels.isotropic import RBF, Exponential, Isotropy, Matern32, Matern52, RationalQuadratic
from pyro.contrib.gp.kernels.kernel import (Combination, Exponent, Kernel, Product, Sum, Transforming, VerticalScaling,
                                            Warping)
from pyro.contrib.gp.kernels.periodic import Cosine, Periodic
from pyro.contrib.gp.kernels.static import Constant, WhiteNoise

__all__ = [
    "Kernel",
    "Brownian",
    "Constant",
    "Coregionalize",
    "Cosine",
    "DotProduct",
    "Exponent",
    "Exponential",
    "Isotropy",
    "RationalQuadratic",
    "Linear",
    "Matern32",
    "Matern52",
    "Periodic",
    "Polynomial",
    "RBF",
    "WhiteNoise",
    "Combination",
    "Sum",
    "Product",
    "Transforming",
    "VerticalScaling",
    "Warping",
]

# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: pyro.contrib.gp.kernels.{0}
        :members:
        :undoc-members:
        :special-members: __call__
        :show-inheritance:
        :member-order: bysource
    '''.format(_name)
    for _name in sorted(__all__)
])
