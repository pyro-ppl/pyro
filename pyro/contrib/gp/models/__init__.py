from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.models.gplvm import GPLVM
from pyro.contrib.gp.models.gpr import GPRegression
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.models.sgpr import SparseGPRegression
from pyro.contrib.gp.models.vgp import VariationalGP
from pyro.contrib.gp.models.vsgp import VariationalSparseGP

__all__ = [
    "GPLVM",
    "GPModel",
    "GPRegression",
    "SparseGPRegression",
    "VariationalGP",
    "VariationalSparseGP",
]
