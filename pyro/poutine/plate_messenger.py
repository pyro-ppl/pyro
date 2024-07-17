# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Iterator, Optional

import pyro
from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.messenger import Messenger, block_messengers
from pyro.poutine.subsample_messenger import SubsampleMessenger

if TYPE_CHECKING:
    import torch

    from pyro.poutine.runtime import Message


class PlateMessenger(SubsampleMessenger):
    """
    Swiss army knife of broadcasting amazingness:
    combines shape inference, independence annotation, and subsampling
    """

    def _process_message(self, msg: "Message") -> None:
        super()._process_message(msg)
        BroadcastMessenger._pyro_sample(msg)

    def __enter__(self) -> Optional["torch.Tensor"]:  # type: ignore[override]
        super().__enter__()
        if self._vectorized and self._indices is not None:
            return self.indices
        return None


@contextmanager
def block_plate(
    name: Optional[str] = None, dim: Optional[int] = None, *, strict: bool = True
) -> Iterator[None]:
    """
    EXPERIMENTAL Context manager to temporarily block a single enclosing plate.

    This is useful for sampling auxiliary variables or lazily sampling global
    variables that are needed in a plated context. For example the following
    models are equivalent:

    Example::

        def model_1(data):
            loc = pyro.sample("loc", dist.Normal(0, 1))
            with pyro.plate("data", len(data)):
                with block_plate("data"):
                    scale = pyro.sample("scale", dist.LogNormal(0, 1))
                pyro.sample("x", dist.Normal(loc, scale))

        def model_2(data):
            loc = pyro.sample("loc", dist.Normal(0, 1))
            scale = pyro.sample("scale", dist.LogNormal(0, 1))
            with pyro.plate("data", len(data)):
                pyro.sample("x", dist.Normal(loc, scale))

    :param str name: Optional name of plate to match.
    :param int dim: Optional dim of plate to match. Must be negative.
    :param bool strict: Whether to error if no matching plate is found.
        Defaults to True.
    :raises: ValueError if no enclosing plate was found and ``strict=True``.
    """
    if (name is not None) == (dim is not None):
        raise ValueError("Exactly one of name,dim must be specified")
    if name is not None:
        assert isinstance(name, str)
    if dim is not None:
        assert isinstance(dim, int)
        assert dim < 0

    def predicate(messenger: Messenger) -> bool:
        if not isinstance(messenger, PlateMessenger):
            return False
        if name is not None:
            return messenger.name == name
        if dim is not None:
            return messenger.dim == dim
        raise ValueError("Unreachable")

    with block_messengers(predicate) as matches:
        if strict and len(matches) != 1:
            raise ValueError(
                f"block_plate matched {len(matches)} messengers. "
                "Try either removing the block_plate or "
                "setting strict=False."
            )
        yield


class PyroSamplePlateScope(Messenger):
    """
    Handler for executing PyroSample statements in a more intuitive plate context.
    """
    def __init__(self, allowed_plates: Iterable[str] = ()):
        self._inner_allowed_plates = frozenset(allowed_plates)

    def __enter__(self):
        self._plates: frozenset[str] = frozenset(p.name for p in pyro.poutine.runtime.get_plates()) | self._inner_allowed_plates
        return super().__enter__()

    def _is_local_plate(self, m: Messenger) -> bool:
        return isinstance(m, PlateMessenger) and m.name not in self._plates

    def _pyro_sample(self, msg):
        if not msg["infer"].get("_original_pyrosample_dist", None):
            return
        msg["stop"] = True
        msg["done"] = True
        with block_messengers(lambda m: m is self or self._is_local_plate(m)):
            d = msg["infer"].pop("_original_pyrosample_dist")
            msg["value"] = pyro.sample(msg["name"], d, obs=msg["value"] if msg["is_observed"] else None, infer=msg["infer"])
