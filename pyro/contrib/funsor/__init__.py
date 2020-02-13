# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro.poutine.runtime

from .named_messenger import named


@pyro.poutine.runtime.effectful(type="to_funsor")
def to_funsor(x, output=None, dim_to_name=None):
    import funsor
    return funsor.to_funsor(x, output=output, dim_to_name=dim_to_name)


@pyro.poutine.runtime.effectful(type="to_data")
def to_data(x, name_to_dim=None):
    import funsor
    return funsor.to_data(x, name_to_dim=name_to_dim)


__all__ = [
    "to_data",
    "to_funsor",
    "named",
]
