from __future__ import absolute_import, division, print_function

from pyro.infer.abstract_infer import Marginal, TracePosterior
from pyro.infer.elbo import ELBO
from pyro.infer.enum import config_enumerate
from pyro.infer.importance import Importance
from pyro.infer.search import Search
from pyro.infer.svi import SVI
from pyro.infer.advi import ADVI, ADVIMultivariateNormal, ADVIDiagonalNormal

# flake8: noqa

_VALIDATION_ENABLED = False


def enable_validation(is_validate):
    global _VALIDATION_ENABLED
    _VALIDATION_ENABLED = is_validate


def is_validation_enabled():
    return _VALIDATION_ENABLED
