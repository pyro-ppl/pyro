# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.distributions import Categorical
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings

from .messenger import Messenger
from .runtime import _ENUM_ALLOCATOR


def _tmc_mixture_sample(msg):
    dist, num_samples = msg["fn"], msg["infer"].get("num_samples")

    # find batch dims that aren't plate dims
    batch_shape = [1] * len(dist.batch_shape)
    for f in msg["cond_indep_stack"]:
        if f.vectorized:
            batch_shape[f.dim] = f.size if f.size > 0 else dist.batch_shape[f.dim]
    batch_shape = tuple(batch_shape)

    # sample a batch
    sample_shape = (num_samples,)
    fat_sample = dist(sample_shape=sample_shape)  # TODO thin before sampling
    assert fat_sample.shape == sample_shape + dist.batch_shape + dist.event_shape
    assert any(d > 1 for d in fat_sample.shape)

    target_shape = (num_samples,) + batch_shape + dist.event_shape

    # if this site has any possible ancestors, sample ancestor indices uniformly
    thin_sample = fat_sample
    if thin_sample.shape != target_shape:

        index = [Ellipsis] + [slice(None)] * (len(thin_sample.shape) - 1)
        squashed_dims = []
        for squashed_dim, squashed_size in zip(range(1, len(thin_sample.shape)), thin_sample.shape[1:]):
            if squashed_size > 1 and (target_shape[squashed_dim] == 1 or squashed_dim == 0):
                # uniformly sample one ancestor per upstream particle population
                ancestor_dist = Categorical(logits=torch.zeros((squashed_size,), device=thin_sample.device))
                ancestor_index = ancestor_dist.sample(sample_shape=(num_samples,))
                index[squashed_dim] = ancestor_index
                squashed_dims.append(squashed_dim)

        thin_sample = Vindex(thin_sample)[tuple(index)]
        for squashed_dim in squashed_dims:
            thin_sample = thin_sample.unsqueeze(squashed_dim)

    assert thin_sample.shape == target_shape
    return thin_sample


def _tmc_diagonal_sample(msg):
    dist, num_samples = msg["fn"], msg["infer"].get("num_samples")

    # find batch dims that aren't plate dims
    batch_shape = [1] * len(dist.batch_shape)
    for f in msg["cond_indep_stack"]:
        if f.vectorized:
            batch_shape[f.dim] = f.size if f.size > 0 else dist.batch_shape[f.dim]
    batch_shape = tuple(batch_shape)

    # sample a batch
    sample_shape = (num_samples,)
    fat_sample = dist(sample_shape=sample_shape)  # TODO thin before sampling
    assert fat_sample.shape == sample_shape + dist.batch_shape + dist.event_shape
    assert any(d > 1 for d in fat_sample.shape)

    target_shape = (num_samples,) + batch_shape + dist.event_shape

    # if this site has any ancestors, choose ancestors from diagonal approximation
    thin_sample = fat_sample
    if thin_sample.shape != target_shape:

        index = [Ellipsis] + [slice(None)] * (len(thin_sample.shape) - 1)
        squashed_dims = []
        for squashed_dim, squashed_size in zip(range(1, len(thin_sample.shape)), thin_sample.shape[1:]):
            if squashed_size > 1 and (target_shape[squashed_dim] == 1 or squashed_dim == 0):
                # diagonal approximation: identify particle indices across populations
                ancestor_index = torch.arange(squashed_size, device=thin_sample.device)
                index[squashed_dim] = ancestor_index
                squashed_dims.append(squashed_dim)

        thin_sample = Vindex(thin_sample)[tuple(index)]
        for squashed_dim in squashed_dims:
            thin_sample = thin_sample.unsqueeze(squashed_dim)

    assert thin_sample.shape == target_shape
    return thin_sample


def enumerate_site(msg):
    dist = msg["fn"]
    num_samples = msg["infer"].get("num_samples", None)
    if num_samples is None:
        # Enumerate over the support of the distribution.
        value = dist.enumerate_support(expand=msg["infer"].get("expand", False))
    elif num_samples > 1 and not msg["infer"].get("expand", False):
        tmc_strategy = msg["infer"].get("tmc", "diagonal")
        if tmc_strategy == "mixture":
            value = _tmc_mixture_sample(msg)
        elif tmc_strategy == "diagonal":
            value = _tmc_diagonal_sample(msg)
        else:
            raise ValueError("{} not a valid TMC strategy".format(tmc_strategy))
    elif num_samples > 1 and msg["infer"]["expand"]:
        # Monte Carlo sample the distribution.
        value = dist(sample_shape=(num_samples,))
    assert value.dim() == 1 + len(dist.batch_shape) + len(dist.event_shape)
    return value


class EnumMessenger(Messenger):
    """
    Enumerates in parallel over discrete sample sites marked
    ``infer={"enumerate": "parallel"}``.

    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer or None.
    """
    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        super().__init__()

    def __enter__(self):
        if self.first_available_dim is not None:
            _ENUM_ALLOCATOR.set_first_available_dim(self.first_available_dim)
        self._markov_depths = {}  # site name -> depth (nonnegative integer)
        self._param_dims = {}  # site name -> (enum dim -> unique id)
        self._value_dims = {}  # site name -> (enum dim -> unique id)
        return super().__enter__()

    @ignore_jit_warnings()
    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"] or not isinstance(msg["fn"], TorchDistributionMixin):
            return

        # Compute upstream dims in scope; these are unsafe to use for this site's target_dim.
        scope = msg["infer"].get("_markov_scope")  # site name -> markov depth
        param_dims = _ENUM_ALLOCATOR.dim_to_id.copy()  # enum dim -> unique id
        if scope is not None:
            for name, depth in scope.items():
                if self._markov_depths[name] == depth:  # hide sites whose markov context has exited
                    param_dims.update(self._value_dims[name])
            self._markov_depths[msg["name"]] = msg["infer"]["_markov_depth"]
        self._param_dims[msg["name"]] = param_dims
        if msg["is_observed"] or msg["infer"].get("enumerate") != "parallel":
            return

        # Compute an enumerated value (at an arbitrary dim).
        value = enumerate_site(msg)
        actual_dim = -1 - len(msg["fn"].batch_shape)  # the leftmost dim of log_prob

        # Move actual_dim to a safe target_dim.
        target_dim, id_ = _ENUM_ALLOCATOR.allocate(None if scope is None else param_dims)
        event_dim = msg["fn"].event_dim
        categorical_support = getattr(value, '_pyro_categorical_support', None)
        if categorical_support is not None:
            # Preserve categorical supports to speed up Categorical.log_prob().
            # See pyro/distributions/torch.py for details.
            assert target_dim < 0
            value = value.reshape(value.shape[:1] + (1,) * (-1 - target_dim))
            value._pyro_categorical_support = categorical_support
        elif actual_dim < target_dim:
            assert value.size(target_dim - event_dim) == 1, \
                'pyro.markov dim conflict at dim {}'.format(actual_dim)
            value = value.transpose(target_dim - event_dim, actual_dim - event_dim)
            while value.dim() and value.size(0) == 1:
                value = value.squeeze(0)
        elif target_dim < actual_dim:
            diff = actual_dim - target_dim
            value = value.reshape(value.shape[:1] + (1,) * diff + value.shape[1:])

        # Compute dims passed downstream through the value.
        value_dims = {dim: param_dims[dim] for dim in range(event_dim - value.dim(), 0)
                      if value.size(dim - event_dim) > 1 and dim in param_dims}
        value_dims[target_dim] = id_

        msg["infer"]["_enumerate_dim"] = target_dim
        msg["infer"]["_dim_to_id"] = value_dims
        msg["value"] = value
        msg["done"] = True

    def _pyro_post_sample(self, msg):
        # Save all dims exposed in this sample value.
        # Whereas all of site["_dim_to_id"] are needed to interpret a
        # site's log_prob tensor, only a filtered subset self._value_dims[msg["name"]]
        # are needed to interpret a site's value.
        if not isinstance(msg["fn"], TorchDistributionMixin):
            return
        value = msg["value"]
        if value is None:
            return
        shape = value.data.shape[:value.dim() - msg["fn"].event_dim]
        dim_to_id = msg["infer"].setdefault("_dim_to_id", {})
        dim_to_id.update(self._param_dims.get(msg["name"], {}))
        with ignore_jit_warnings():
            self._value_dims[msg["name"]] = {dim: id_ for dim, id_ in dim_to_id.items()
                                             if len(shape) >= -dim and shape[dim] > 1}
