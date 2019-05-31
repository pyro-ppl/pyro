from __future__ import absolute_import, division, print_function

import re
import weakref
from abc import ABCMeta, abstractmethod

import torch
from contextlib2 import ExitStack
from six import add_metaclass
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.poutine.runtime as runtime
from pyro.contrib.autoguide.initialization import InitMessenger
from pyro.distributions.util import broadcast_shape, sum_rightmost
from pyro.poutine.util import prune_subsample_sites


@add_metaclass(ABCMeta)
class EasyGuide(object):
    """
    Base class for "easy guides".

    Derived classes should define a :meth:`guide` method. This :meth:`guide`
    method can combine ordinary guide statements (e.g. ``pyro.sample`` and
    ``pyro.param``) with the following special statements:

    - ``group = self.group(...)`` selects multiple ``pyro.sample`` sites in the
      model. See :class:`Group` for subsequent methods.
    - ``with self.plate(...): ...`` should be used instead of ``pyro.plate``.
    - ``self.map_estimate(...)`` uses a ``Delta`` guide for a single site.

    Derived classes may also override the :meth:`init` method to provide custom
    initialization for models sites.

    :param callable model: A Pyro model.
    """
    def __init__(self, model):
        self.model = model
        self.prototype_trace = None
        self.frames = {}
        self.groups = {}
        self.plates = {}

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        model = InitMessenger(self.init)(self.model)
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            for frame in site["cond_indep_stack"]:
                if not frame.vectorized:
                    raise NotImplementedError("EasyGuide does not support sequential pyro.plate")
                self.frames[frame.name] = frame

    @abstractmethod
    def guide(self, *args, **kargs):
        """
        Guide implementation, to be overridden by user.
        """
        raise NotImplementedError

    def init(self, site):
        """
        Model initialization method, may be overridden by user.

        This should input a site and output a valid sample from that site.
        The default behavior is to draw a random sample::

            return site["fn"]()

        For other possible initialization functions see
        http://docs.pyro.ai/en/stable/contrib.autoguide.html#module-pyro.contrib.autoguide.initialization
        """
        return site["fn"]()

    def __call__(self, *args, **kwargs):
        """
        Runs the guide. This is typically used by inference algorithms.
        """
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)
        result = self.guide(*args, **kwargs)
        self.plates.clear()
        return result

    def plate(self, name, size=None, subsample_size=None, subsample=None, *args, **kwargs):
        """
        A wrapper around :class:`pyro.plate` to allow `EasyGuide` to
        automatically construct plates. You should use this rather than
        :class:`pyro.plate` inside your :meth:`guide` implementation.
        """
        if name not in self.plates:
            self.plates[name] = pyro.plate(name, size, subsample_size, subsample, *args, **kwargs)
        return self.plates[name]

    def group(self, match=".*"):
        """
        Select a :class:`Group` of model sites for joint guidance.

        :param str match: A regex string matching names of model sample sites.
        :return: A group of model sites.
        :rtype: Group
        """
        if match not in self.groups:
            sites = [site
                     for name, site in self.prototype_trace.iter_stochastic_nodes()
                     if re.match(match, name)]
            if not sites:
                raise ValueError("EasyGuide.group() pattern {} matched no model sites"
                                 .format(repr(match)))
            self.groups[match] = Group(self, sites)
        return self.groups[match]

    def map_estimate(self, name):
        """
        Construct a maximum a posteriori (MAP) guide using Delta distributions.

        :param str name: The name of a model sample site.
        :return: A sampled value.
        :rtype: torch.Tensor
        """
        site = self.prototype_trace.nodes[name]
        fn = site["fn"]
        param_name = "auto_{}".format(name)
        init_value = None
        if param_name not in pyro.get_param_store():
            init_value = site["value"].detach()
        with ExitStack() as stack:
            for frame in site["cond_indep_stack"]:
                plate = self.plate(frame.name)
                if plate not in runtime._PYRO_STACK:
                    stack.enter_context(plate)
                elif init_value is not None and plate.subsample_size < plate.size:
                    # Repeat the init_value to full size.
                    dim = plate.dim - fn.event_dim
                    assert init_value.size(dim) == plate.subsample_size
                    ind = torch.arange(plate.size, device=init_value.device)
                    ind = ind % plate.subsample_size
                    init_value = init_value.index_select(dim, ind)
            value = pyro.param(param_name, init_value,
                               constraint=fn.support, event_dim=fn.event_dim)
            return pyro.sample(name, dist.Delta(value, event_dim=fn.event_dim))


class Group(object):
    """
    An autoguide helper to match a group of model sites.

    :ivar torch.Size event_shape: The total flattened concatenated shape of all
        matching sample sites in the model.
    :ivar list prototype_sites: A list of all matching sample sites in a
        prototype trace of the model.
    :param EasyGuide guide: An easyguide instance.
    :param list sites: A list of model sites.
    """
    def __init__(self, guide, sites):
        assert isinstance(sites, list)
        assert sites
        self._guide = weakref.ref(guide)
        self.prototype_sites = sites
        self._site_sizes = {}
        self._site_batch_shapes = {}

        # A group is in a frame only if all its sample sites are in that frame.
        # Thus a group can be subsampled only if all its sites can be subsampled.
        self.common_frames = frozenset.intersection(*(
            frozenset(f for f in site["cond_indep_stack"] if f.vectorized)
            for site in sites))
        rightmost_common_dim = -float('inf')
        if self.common_frames:
            rightmost_common_dim = max(f.dim for f in self.common_frames)

        # Compute flattened concatenated event_shape and split batch_shape into
        # a common batch_shape (which can change each SVI step due to
        # subsampling) and site batch_shapes (which must remain constant size).
        for site in sites:
            site_event_numel = torch.Size(site["fn"].event_shape).numel()
            site_batch_shape = list(site["fn"].batch_shape)
            for f in self.common_frames:
                # Consider this dim part of the common_batch_shape.
                site_batch_shape[f.dim] = 1
            while site_batch_shape and site_batch_shape[0] == 1:
                site_batch_shape = site_batch_shape[1:]
            if len(site_batch_shape) > -rightmost_common_dim:
                raise ValueError(
                    "Group expects all per-site plates to be right of all common plates, "
                    "but found a per-site plate {} on left at site {}"
                    .format(-len(site_batch_shape), repr(site["name"])))
            site_batch_shape = torch.Size(site_batch_shape)
            self._site_batch_shapes[site["name"]] = site_batch_shape
            self._site_sizes[site["name"]] = site_batch_shape.numel() * site_event_numel
        self.event_shape = torch.Size([sum(self._site_sizes.values())])

    @property
    def guide(self):
        return self._guide()

    def sample(self, guide_name, fn, infer=None):
        """
        Wrapper around ``pyro.sample()`` to create a single auxiliary sample
        site and then unpack to multiple sample sites for model replay.

        :param str guide_name: The name of the auxiliary guide site.
        :param callable fn: A distribution with shape ``self.event_shape``.
        :param dict infer: Optional inference configuration dict.
        :returns: A pair ``(guide_z, model_zs)`` where ``guide_z`` is the
            single concatenated blob and ``model_zs`` is a dict mapping
            site name to constrained model sample.
        :rtype: tuple
        """
        # Sample a packed tensor.
        if fn.event_shape != self.event_shape:
            raise ValueError("Invalid fn.event_shape for group: expected {}, actual {}"
                             .format(tuple(self.event_shape), tuple(fn.event_shape)))
        if infer is None:
            infer = {}
        infer["is_auxiliary"] = True
        guide_z = pyro.sample(guide_name, fn, infer=infer)
        common_batch_shape = guide_z.shape[:-1]

        model_zs = {}
        pos = 0
        for site in self.prototype_sites:
            name = site["name"]
            fn = site["fn"]

            # Extract slice from packed sample.
            size = self._site_sizes[name]
            batch_shape = broadcast_shape(common_batch_shape, self._site_batch_shapes[name])
            unconstrained_z = guide_z[..., pos: pos + size]
            unconstrained_z = unconstrained_z.reshape(batch_shape + fn.event_shape)
            pos += size

            # Transform to constrained space.
            transform = biject_to(fn.support)
            z = transform(unconstrained_z)
            log_density = transform.inv.log_abs_det_jacobian(z, unconstrained_z)
            log_density = sum_rightmost(log_density, log_density.dim() - z.dim() + fn.event_dim)
            delta_dist = dist.Delta(z, log_density=log_density, event_dim=fn.event_dim)

            # Replay model sample statement.
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    plate = self.guide.plate(frame.name)
                    if plate not in runtime._PYRO_STACK:
                        stack.enter_context(plate)
                model_zs[name] = pyro.sample(name, delta_dist)

        return guide_z, model_zs

    def map_estimate(self):
        """
        Construct a maximum a posteriori (MAP) guide using Delta distributions.

        :return: A dict mapping model site name to sampled value.
        :rtype: dict
        """
        return {site["name"]: self.guide.map_estimate(site["name"])
                for site in self.prototype_sites}


def easy_guide(model):
    """
    Convenience decorator to create an :class:`EasyGuide` .
    The following are equivalent::

        # Version 1. Decorate a function.
        @easy_guide(model)
        def guide(self, foo, bar):
            return my_guide(foo, bar)

        # Version 2. Create and instantiate a subclass of EasyGuide.
        class Guide(EasyGuide):
            def guide(self, foo, bar):
                return my_guide(foo, bar)
        guide = Guide(model)

    :param callable model: a Pyro model.
    """

    def decorator(fn):
        Guide = type(fn.__name__, (EasyGuide,), {"guide": fn})
        return Guide(model)

    return decorator
