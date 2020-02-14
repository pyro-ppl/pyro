# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro.poutine.runtime

from .named_messenger import _DIM_STACK, named


@pyro.poutine.runtime.effectful(type="to_funsor")
def to_funsor(x, output=None, dim_to_name=None):
    if pyro.poutine.runtime.am_i_wrapped() and not dim_to_name:
        dim_to_name = _DIM_STACK.global_frame.dim_to_name.copy()
    import funsor
    return funsor.to_funsor(x, output=output, dim_to_name=dim_to_name)


@pyro.poutine.runtime.effectful(type="to_data")
def to_data(x, name_to_dim=None):
    if pyro.poutine.runtime.am_i_wrapped() and not name_to_dim:
        name_to_dim = _DIM_STACK.global_frame.name_to_dim.copy()
    import funsor
    return funsor.to_data(x, name_to_dim=name_to_dim)


__all__ = [
    "to_data",
    "to_funsor",
    "named",
]
