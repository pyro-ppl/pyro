from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from .delta import Delta
from .util import is_identically_zero


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

    def get_log_importance(self, fn, values, value):
        r"""
        Get log importance weight :math:`log \frac {p(x)}{q(X)}` if any.
        Defaults to zero.
        """
        return 0.0

    def get_value_and_fn(self, fn, values):
        value = self.transform_values(fn, values)
        importance = self.get_log_importance(fn, values, value)
        new_fn = Delta(value, event_dim=fn.event_dim, log_density=importance)
        if is_identically_zero(importance):
            new_fn = new_fn.mask(False)
        return value, new_fn


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


class TransformReparameterizer(Reparameterizer):
    """
    Arbitrary transform reparameterizer.

    This can be used to reparameterize wrt an arbitrary bijective
    :class:`~torch.distributions.transforms.Transform` object, and requires
    only the forward ``.__call__()`` method and the ``.log_abs_det_jacobian()``
    transform to be defined, as in [1].

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704

    :param ~pyro.distributions.TorchDistribution base_dist: A base
        distribution for the auxiliary latent variable.
    :param ~torch.distributions.transforms.Transform transform: A bijective
        transform defining forward and log abs det jacobian methods.
    """
    def __init__(self, base_dist, transform):
        super().__init__()
        self.base_dist = base_dist
        self.transform = transform

    def get_dists(self, fn):
        return OrderedDict(base_dist=self.base_dist)

    def transform_values(self, fn, values):
        return self.transform(values["base_dist"])

    def get_log_importance(self, fn, values, value):
        z = values["base_dist"]
        x = value
        return self.fn.log_prob(x) + self.base_dist.log_abs_det_jacobian(z, x)
