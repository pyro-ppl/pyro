# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy
import pytest
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.contrib.autoname.named as named
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import HMC, NUTS
from pyro.params import param_with_module_name


# Fix seed for all doctest runs.
pyro.set_rng_seed(0)


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    doctest_namespace['dist'] = dist
    doctest_namespace['gp'] = gp
    doctest_namespace['named'] = named
    doctest_namespace['np'] = numpy
    doctest_namespace['param_with_module_name'] = param_with_module_name
    doctest_namespace['poutine'] = poutine
    doctest_namespace['pyro'] = pyro
    doctest_namespace['torch'] = torch
    doctest_namespace['EmpiricalMarginal'] = EmpiricalMarginal
    doctest_namespace['HMC'] = HMC
    doctest_namespace['MCMC'] = MCMC
    doctest_namespace['NUTS'] = NUTS
