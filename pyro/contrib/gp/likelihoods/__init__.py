# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pyro.contrib.gp.likelihoods.binary import Binary
from pyro.contrib.gp.likelihoods.gaussian import Gaussian
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.contrib.gp.likelihoods.multi_class import MultiClass
from pyro.contrib.gp.likelihoods.poisson import Poisson

__all__ = [
    "Likelihood",
    "Binary",
    "Gaussian",
    "MultiClass",
    "Poisson",
]


# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: pyro.contrib.gp.likelihoods.{0}
        :members:
        :undoc-members:
        :special-members: __call__
        :show-inheritance:
        :member-order: bysource
    '''.format(_name)
    for _name in __all__
])
