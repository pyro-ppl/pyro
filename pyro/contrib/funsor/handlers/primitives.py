# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro.poutine.runtime

from pyro.contrib.funsor.handlers.runtime import _DIM_STACK, DimRequest, DimType


@pyro.poutine.runtime.effectful(type="to_funsor")
def to_funsor(x, output=None, dim_to_name=None, dim_type=DimType.LOCAL):
    import funsor
    if pyro.poutine.runtime.am_i_wrapped() and not dim_to_name:
        dim_to_name = _DIM_STACK.global_frame.dim_to_name.copy()
    assert not dim_to_name or not any(isinstance(name, DimRequest) for name in dim_to_name.values())
    return funsor.to_funsor(x, output=output, dim_to_name=dim_to_name)


@pyro.poutine.runtime.effectful(type="to_data")
def to_data(x, name_to_dim=None, dim_type=DimType.LOCAL):
    import funsor
    if pyro.poutine.runtime.am_i_wrapped() and not name_to_dim:
        name_to_dim = _DIM_STACK.global_frame.name_to_dim.copy()
    assert not name_to_dim or not any(isinstance(dim, DimRequest) for dim in name_to_dim.values())
    return funsor.to_data(x, name_to_dim=name_to_dim)
