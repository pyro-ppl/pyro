# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

from .torch import MultivariateNormal, Normal


class NanMaskedNormal(Normal):
    """
    Wrapper around :class:`~pyro.distributions.Normal` to allow partially
    observed data as specified by NAN elements in :meth:`log_prob`; the
    ``log_prob`` of these elements will be zero. This is useful for likelihoods
    with missing data.

    Example::

        from math import nan
        data = torch.tensor([0.5, 0.1, nan, 0.9])
        with pyro.plate("data", len(data)):
            pyro.sample("obs", NanMaskedNormal(0, 1), obs=data)
    """

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        ok = value.isfinite()
        if ok.all():
            return super().log_prob(value)

        # Broadcast all tensors.
        value, ok, loc, scale = torch.broadcast_tensors(value, ok, self.loc, self.scale)
        result = value.new_zeros(value.shape)

        # Evaluate ok elements.
        if ok.any():
            marginal = Normal(loc[ok], scale[ok], validate_args=False)
            result[ok] = marginal.log_prob(value[ok])
        return result


class NanMaskedMultivariateNormal(MultivariateNormal):
    """
    Wrapper around :class:`~pyro.distributions.MultivariateNormal` to allow
    partially observed data as specified by NAN elements in the argument to
    :meth:`log_prob`. The ``log_prob`` of these events will marginalize over
    the NAN elements. This is useful for likelihoods with missing data.

    Example::

        from math import nan
        data = torch.tensor([
            [0.1, 0.2, 3.4],
            [0.5, 0.1, nan],
            [0.6, nan, nan],
            [nan, 0.5, nan],
            [nan, nan, nan],
        ])
        with pyro.plate("data", len(data)):
            pyro.sample(
                "obs",
                NanMaskedMultivariateNormal(torch.zeros(3), torch.eye(3)),
                obs=data,
            )
    """

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        ok = value.isfinite()
        if ok.all():
            return super().log_prob(value)

        # Broadcast all tensors. This might waste some computation by eagerly
        # broadcasting, but the optimal implementation is quite complex.
        value, ok, loc = torch.broadcast_tensors(value, ok, self.loc)
        cov = self.covariance_matrix.expand(loc.shape + loc.shape[-1:])

        # Flatten.
        result_shape = value.shape[:-1]
        n = result_shape.numel()
        p = value.shape[-1]
        value = value.reshape(n, p)
        ok = ok.reshape(n, p)
        loc = loc.reshape(n, p)
        cov = cov.reshape(n, p, p)
        result = value.new_zeros(n)

        # Evaluate ok elements.
        for pattern in sorted(set(map(tuple, ok.tolist()))):
            if not any(pattern):
                continue
            # Marginalize out NAN elements.
            col_mask = torch.tensor(pattern)
            row_mask = (ok == col_mask).all(-1)
            ok_value = value[row_mask][:, col_mask]
            ok_loc = loc[row_mask][:, col_mask]
            ok_cov = cov[row_mask][:, col_mask][:, :, col_mask]
            marginal = MultivariateNormal(ok_loc, ok_cov, validate_args=False)
            result[row_mask] = marginal.log_prob(ok_value)

        # Unflatten.
        return result.reshape(result_shape)
