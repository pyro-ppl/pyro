# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

# Import * to get the latest upstream constraints.
from torch.distributions.constraints import *  # noqa F403

# Additionally try to import explicitly to help mypy static analysis.
try:
    from torch.distributions.constraints import (
        Constraint,
        boolean,
        cat,
        corr_cholesky,
        dependent,
        dependent_property,
        greater_than,
        greater_than_eq,
        half_open_interval,
        independent,
        integer_interval,
        interval,
        is_dependent,
        less_than,
        lower_cholesky,
        lower_triangular,
        multinomial,
        nonnegative,
        nonnegative_integer,
        one_hot,
        positive,
        positive_definite,
        positive_integer,
        positive_semidefinite,
        real,
        real_vector,
        simplex,
        square,
        stack,
        symmetric,
        unit_interval,
    )
except ImportError:
    pass

import torch
from torch.distributions.constraints import __all__ as torch_constraints


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


class _UnitLowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with all ones diagonals.
    """

    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        lower_triangular = (
            (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]
        )

        ones_diagonal = (value.diagonal(dim1=-2, dim2=-1) == 1).min(-1)[0]
        return lower_triangular & ones_diagonal


corr_matrix = _CorrMatrix()
integer = _Integer()
ordered_vector = _OrderedVector()
positive_ordered_vector = _PositiveOrderedVector()
sphere = _Sphere()
softplus_positive = _SoftplusPositive()
softplus_lower_cholesky = _SoftplusLowerCholesky()
unit_lower_cholesky = _UnitLowerCholesky()
corr_cholesky_constraint = corr_cholesky  # noqa: F405 DEPRECATED

__all__ = [
    "Constraint",
    "boolean",
    "cat",
    "corr_cholesky",
    "corr_cholesky_constraint",
    "corr_matrix",
    "dependent",
    "dependent_property",
    "greater_than",
    "greater_than_eq",
    "half_open_interval",
    "independent",
    "integer",
    "integer_interval",
    "interval",
    "is_dependent",
    "less_than",
    "lower_cholesky",
    "lower_triangular",
    "multinomial",
    "nonnegative",
    "nonnegative_integer",
    "one_hot",
    "ordered_vector",
    "positive",
    "positive_definite",
    "positive_integer",
    "positive_ordered_vector",
    "positive_semidefinite",
    "real",
    "real_vector",
    "simplex",
    "softplus_lower_cholesky",
    "softplus_positive",
    "sphere",
    "square",
    "stack",
    "symmetric",
    "unit_interval",
    "unit_lower_cholesky",
]

__all__.extend(torch_constraints)
__all__[:] = sorted(set(__all__))
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
            (
                "alias of :class:`torch.distributions.constraints.{}`".format(_name)
                if globals()[_name].__module__.startswith("torch")
                else ".. autoclass:: {}".format(
                    _name
                    if type(globals()[_name]) is type
                    else type(globals()[_name]).__name__
                )
            ),
        )
        for _name in sorted(__all__)
    ]
)
