# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import funsor

import pyro.distributions
import pyro.poutine.runtime
import pyro.primitives
from pyro.contrib.funsor.handlers.runtime import _DIM_STACK, DimType


@pyro.poutine.runtime.effectful(type="to_funsor")
def to_funsor(x, output=None, dim_to_name=None, dim_type=DimType.LOCAL):
    if pyro.poutine.runtime.am_i_wrapped() and not dim_to_name:
        dim_to_name = _DIM_STACK.global_frame.dim_to_name.copy()
    return funsor.to_funsor(x, output=output, dim_to_name=dim_to_name)


@pyro.poutine.runtime.effectful(type="to_data")
def to_data(x, name_to_dim=None, dim_type=DimType.LOCAL):
    if pyro.poutine.runtime.am_i_wrapped() and not name_to_dim:
        name_to_dim = _DIM_STACK.global_frame.name_to_dim.copy()
    return funsor.to_data(x, name_to_dim=name_to_dim)


class _EmptyDist(pyro.distributions.Distribution):
    def log_prob(self, value):
        raise NotImplementedError("Should not be here. Cannot score from null distribution")

    def sample(self):
        raise NotImplementedError("Should not be here. Cannot sample from null distribution")


@functools.wraps(pyro.primitives.sample)
def sample(name, fn=None, *args, **kwargs):
    fn = fn if fn is not None else _EmptyDist()
    return pyro.primitives.sample(name, fn, *args, **kwargs)
