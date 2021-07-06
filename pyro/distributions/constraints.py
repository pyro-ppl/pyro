# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions.constraints import *  # noqa F403

# isort: split

import torch
from torch.distributions.constraints import Constraint
from torch.distributions.constraints import __all__ as torch_constraints
from torch.distributions.constraints import (
    independent,
    lower_cholesky,
    positive,
    positive_definite,
)


# TODO move this upstream to torch.distributions
class _Integer(Constraint):
    """
    Constrain to integers.
    """

    is_discrete = True

    def check(self, value):
        return value % 1 == 0

    def __repr__(self):
        return self.__class__.__name__[1:]


class _Sphere(Constraint):
    """
    Constrain to the Euclidean sphere of any dimension.
    """

    event_dim = 1
    reltol = 10.0  # Relative to finfo.eps.

    def check(self, value):
        eps = torch.finfo(value.dtype).eps
        norm = torch.linalg.norm(value, dim=-1)
        error = (norm - 1).abs()
        return error < self.reltol * eps * value.size(-1) ** 0.5

    def __repr__(self):
        return self.__class__.__name__[1:]


class _CorrMatrix(Constraint):
    """
    Constrains to a correlation matrix.
    """

    event_dim = 2

    def check(self, value):
        # check for diagonal equal to 1
        unit_variance = torch.all(
            torch.abs(torch.diagonal(value, dim1=-2, dim2=-1) - 1) < 1e-6, dim=-1
        )
        # TODO: fix upstream - positive_definite has an extra dimension in front of output shape
        return positive_definite.check(value) & unit_variance


class _OrderedVector(Constraint):
    """
    Constrains to a real-valued tensor where the elements are monotonically
    increasing along the `event_shape` dimension.
    """

    event_dim = 1

    def check(self, value):
        if value.ndim == 0:
            return torch.tensor(False, device=value.device)
        elif value.shape[-1] == 1:
            return torch.ones_like(value[..., 0], dtype=bool)
        else:
            return torch.all(value[..., 1:] > value[..., :-1], dim=-1)


class _PositiveOrderedVector(Constraint):
    """
    Constrains to a positive real-valued tensor where the elements are monotonically
    increasing along the `event_shape` dimension.
    """

    def check(self, value):
        return ordered_vector.check(value) & independent(positive, 1).check(value)


class _SoftplusPositive(type(positive)):
    def __init__(self):
        super().__init__(lower_bound=0.0)


class _SoftplusLowerCholesky(type(lower_cholesky)):
    pass


corr_matrix = _CorrMatrix()
integer = _Integer()
ordered_vector = _OrderedVector()
positive_ordered_vector = _PositiveOrderedVector()
sphere = _Sphere()
softplus_positive = _SoftplusPositive()
softplus_lower_cholesky = _SoftplusLowerCholesky()
corr_cholesky_constraint = corr_cholesky  # noqa: F405 DEPRECATED

__all__ = [
    "corr_cholesky_constraint",
    "corr_matrix",
    "integer",
    "ordered_vector",
    "positive_ordered_vector",
    "softplus_lower_cholesky",
    "softplus_positive",
    "sphere",
]

__all__.extend(torch_constraints)
__all__ = sorted(set(__all__))
del torch_constraints


# Create sphinx documentation.
__doc__ = """
    Pyro's constraints library extends
    :mod:`torch.distributions.constraints`.
"""
__doc__ += "\n".join(
    [
        """
    {}
    ----------------------------------------------------------------
    {}
    """.format(
            _name,
            "alias of :class:`torch.distributions.constraints.{}`".format(_name)
            if globals()[_name].__module__.startswith("torch")
            else ".. autoclass:: {}".format(
                _name
                if type(globals()[_name]) is type
                else type(globals()[_name]).__name__
            ),
        )
        for _name in sorted(__all__)
    ]
)
