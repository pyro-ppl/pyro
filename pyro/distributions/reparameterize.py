from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from pyro.distributions.util import is_identically_one, is_identically_zero, is_validation_enabled


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

    To trigger use in Pyro models use the
    :func:`~pyro.poutine.handlers.reparam` handler.

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
    Generic decentering reparameterizer for distributions that are specified by
    parameters ``loc`` and ``scale`` (and possibly additional
    ``shape_params``). This can be combined with :func:`pyro.param` to learn a
    centering transform::

        x_centered = pyro.param("x_centered", 0.5,
                                constraint=constraints.unit_interval)
        pyro.sample("x", dist.StudentT(df, loc, scale),
                    infer={"reparam": LocScaleReparameterizer(x_centered,
                                                              shape_params=["df"])})

    :param float centered: in ``[0,1]``. If 0 (default), fully decenter the
        distribution; if 1, preserve the centered distribution unchanged.
    :param shape_params: list of additional parameter names to copy unchanged from
        the centered to decentered distribution.
    :type shape_params: tuple or list
    """
    def __init__(self, centered=0.0, shape_params=()):
        assert isinstance(centered, (float, torch.Tensor))
        assert isinstance(shape_params, (tuple, list))
        assert all(isinstance(name, str) for name in shape_params)
        if is_validation_enabled():
            if isinstance(centered, float):
                assert 0 <= centered and centered <= 1
            else:
                assert (0 <= centered).all()
                assert (centered <= 1).all()
        self.centered = centered
        self.shape_params = shape_params

    def get_dists(self, fn):
        if is_identically_one(self.centered):
            return OrderedDict()
        if is_identically_zero(self.centered):
            loc = torch.zeros_like(fn.loc)
            scale = torch.ones_like(fn.scale)
        else:
            loc = fn.loc * self.centered
            scale = fn.scale ** self.centered
        new_fn = type(fn)(loc=loc, scale=scale)
        return OrderedDict([("decentered", new_fn)])

    def transform_values(self, fn, values):
        if is_identically_one(self.centered):
            return fn
        elif is_identically_zero(self.centered):
            return fn.loc + fn.scale * values["decentered"]
        else:
            delta = values["decentered"] - self.centered * fn.loc
            return fn.loc + fn.scale.pow(1 - self.centered) * delta
