# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pyro.util import ignore_jit_warnings
from .messenger import Messenger


class BroadcastMessenger(Messenger):
    """
    Automatically broadcasts the batch shape of the stochastic function
    at a sample site when inside a single or nested plate context.
    The existing `batch_shape` must be broadcastable with the size
    of the :class:`~pyro.plate` contexts installed in the
    `cond_indep_stack`.

    Notice how `model_automatic_broadcast` below automates expanding of
    distribution batch shapes. This makes it easy to modularize a
    Pyro model as the sub-components are agnostic of the wrapping
    :class:`~pyro.plate` contexts.

    >>> def model_broadcast_by_hand():
    ...     with IndepMessenger("batch", 100, dim=-2):
    ...         with IndepMessenger("components", 3, dim=-1):
    ...             sample = pyro.sample("sample", dist.Bernoulli(torch.ones(3) * 0.5)
    ...                                                .expand_by(100))
    ...             assert sample.shape == torch.Size((100, 3))
    ...     return sample

    >>> @poutine.broadcast
    ... def model_automatic_broadcast():
    ...     with IndepMessenger("batch", 100, dim=-2):
    ...         with IndepMessenger("components", 3, dim=-1):
    ...             sample = pyro.sample("sample", dist.Bernoulli(torch.tensor(0.5)))
    ...             assert sample.shape == torch.Size((100, 3))
    ...     return sample
    """

    @staticmethod
    @ignore_jit_warnings(["Converting a tensor to a Python boolean"])
    def _pyro_sample(msg):
        """
        :param msg: current message at a trace site.
        """
        if msg["done"] or msg["type"] != "sample":
            return

        dist = msg["fn"]
        actual_batch_shape = getattr(dist, "batch_shape", None)
        if actual_batch_shape is not None:
            target_batch_shape = [None if size == 1 else size
                                  for size in actual_batch_shape]
            for f in msg["cond_indep_stack"]:
                if f.dim is None or f.size == -1:
                    continue
                assert f.dim < 0
                target_batch_shape = [None] * (-f.dim - len(target_batch_shape)) + target_batch_shape
                if target_batch_shape[f.dim] is not None and target_batch_shape[f.dim] != f.size:
                    raise ValueError("Shape mismatch inside plate('{}') at site {} dim {}, {} vs {}".format(
                        f.name, msg['name'], f.dim, f.size, target_batch_shape[f.dim]))
                target_batch_shape[f.dim] = f.size
            # Starting from the right, if expected size is None at an index,
            # set it to the actual size if it exists, else 1.
            for i in range(-len(target_batch_shape) + 1, 1):
                if target_batch_shape[i] is None:
                    target_batch_shape[i] = actual_batch_shape[i] if len(actual_batch_shape) >= -i else 1
            msg["fn"] = dist.expand(target_batch_shape)
            if msg["fn"].has_rsample != dist.has_rsample:
                msg["fn"].has_rsample = dist.has_rsample  # copy custom attribute
