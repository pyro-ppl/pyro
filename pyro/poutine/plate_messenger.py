# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

from .broadcast_messenger import BroadcastMessenger
from .messenger import block_messengers
from .subsample_messenger import SubsampleMessenger


class PlateMessenger(SubsampleMessenger):
    """
    Swiss army knife of broadcasting amazingness:
    combines shape inference, independence annotation, and subsampling
    """
    def _process_message(self, msg):
        super()._process_message(msg)
        return BroadcastMessenger._pyro_sample(msg)

    def __enter__(self):
        super().__enter__()
        if self._vectorized and self._indices is not None:
            return self.indices
        return None


@contextmanager
def block_plate(name=None, dim=None):
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
    :raises: ValueError if no enclosing plate was found.
    """
    if (name is not None) == (dim is not None):
        raise ValueError("Exactly one of name,dim must be specified")
    if name is not None:
        assert isinstance(name, str)
    if dim is not None:
        assert isinstance(dim, int)
        assert dim < 0

    def predicate(messenger):
        if not isinstance(messenger, PlateMessenger):
            return False
        if name is not None:
            return messenger.name == name
        if dim is not None:
            return messenger.dim == dim

    with block_messengers(predicate) as matches:
        if len(matches) != 1:
            raise ValueError("block_plate matched {} messengers".format(len(matches)))
        yield
