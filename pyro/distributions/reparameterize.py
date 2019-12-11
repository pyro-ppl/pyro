from abc import ABC, abstractmethod
from collections import OrderedDict

import torch


class Reparameterizer(ABC):
    """
    Base class for reparameterization transforms, generalizing [1] to handle
    multiple distributions and auxiliary variables.

    The interface specifies two methods to be used by inference algorithms in
    the following pattern::

        # Transform a site like this...
        pyro.sample("x", dist.Normal(loc, scale))

        # ...to one or more sample sites plus a deterministic site:
        d = dist.Normal(loc, scale)
        values = OrderedDict((name, pyro.sample("x_"+name, fn))
                             for name, fn in reparam.get_dists(d))
        pyro.deterministic("x", reparam.transform_values(d, values))

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf
    """
    @abstractmethod
    def get_dists(self, fn):
        """
        Constructs one or more auxiliary
        :class:`~pyro.distribution.Distribution` s that will replace ``fn``.

        :param ~pyro.distributions.Distribution fn: A base distribution.
        :returns: An OrderedDict mapping name (suffix) to auxiliary distribution.
        :rtype: ~collections.OrderedDict
        """
        raise NotImplementedError

    @abstractmethod
    def transform_values(self, fn, values):
        """
        Deterministically combines samples from the distributions returned by
        :meth:`get_dists` into a single reparameterized sample from ``fn``.

        :param ~pyro.distributions.Distribution fn: A base distribution.
        :param ~collections.OrderedDict values: An OrderedDict mapping name (suffix) to
            sample value drawn from corresponding auxiliary variable.
        :returns: A transformed value.
        :rtype: ~torch.Tensor
        """
        raise NotImplementedError


class TrivialReparameterizer(Reparameterizer):
    """
    Trivial reparameterizer, mainly useful for testing.
    """
    def get_dists(self, fn):
        return OrderedDict([("trivial", fn)])

    def transform_values(self, fn, values):
        return values["trivial"]


class LocScaleReparameterizer(Reparameterizer):
    """
    Generic centering reparameterizer for distributions that are completely
    specified by parameters ``loc`` and ``scale``.
    """
    def get_dists(self, fn):
        loc = torch.zeros_like(fn.loc)
        scale = torch.ones_like(fn.scale)
        new_fn = type(fn)(loc=loc, scale=scale)
        return OrderedDict([("centered", new_fn)])

    def transform_values(self, fn, values):
        return fn.loc + fn.scale * values["centered"]
