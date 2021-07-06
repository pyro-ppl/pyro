# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
The :mod:`pyro.infer.autoguide` module provides algorithms to automatically
generate guides from simple models, for use in :class:`~pyro.infer.svi.SVI`.
For example to generate a mean field Gaussian guide::

    def model():
        ...

    guide = AutoNormal(model)  # a mean field guide
    svi = SVI(model, guide, Adam({'lr': 1e-3}), Trace_ELBO())

Automatic guides can also be combined using :func:`pyro.poutine.block` and
:class:`AutoGuideList`.
"""
import functools
import operator
import warnings
import weakref
from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Callable, Dict, Union

import torch
from torch import nn
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.transforms import affine_autoregressive, iterated
from pyro.distributions.util import eye_like, is_identically_zero, sum_rightmost
from pyro.infer.autoguide.initialization import (
    InitMessenger,
    init_to_feasible,
    init_to_median,
)
from pyro.infer.enum import config_enumerate
from pyro.nn import PyroModule, PyroParam
from pyro.ops.hessian import hessian
from pyro.ops.tensor_utils import periodic_repeat
from pyro.poutine.util import site_is_subsample

from .utils import _product, helpful_support_errors


def _deep_setattr(obj, key, val):
    """
    Set an attribute `key` on the object. If any of the prefix attributes do
    not exist, they are set to :class:`~pyro.nn.PyroModule`.
    """

    def _getattr(obj, attr):
        obj_next = getattr(obj, attr, None)
        if obj_next is not None:
            return obj_next
        setattr(obj, attr, PyroModule())
        return getattr(obj, attr)

    lpart, _, rpart = key.rpartition(".")
    # Recursive getattr while setting any prefix attributes to PyroModule
    if lpart:
        obj = functools.reduce(_getattr, [obj] + lpart.split("."))
    setattr(obj, rpart, val)


def _deep_getattr(obj, key):
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def prototype_hide_fn(msg):
    # Record only stochastic sites in the prototype_trace.
    return msg["type"] != "sample" or msg["is_observed"] or site_is_subsample(msg)


class AutoGuide(PyroModule):
    """
    Base class for automatic guides.

    Derived classes must implement the :meth:`forward` method, with the
    same ``*args, **kwargs`` as the base ``model``.

    Auto guides can be used individually or combined in an
    :class:`AutoGuideList` object.

    :param callable model: A pyro model.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    def __init__(self, model, *, create_plates=None):
        super().__init__(name=type(self).__name__)
        self.master = None
        # Do not register model as submodule
        self._model = (model,)
        self.create_plates = create_plates
        self.prototype_trace = None
        self._prototype_frames = {}

    @property
    def model(self):
        return self._model[0]

    def _update_master(self, master_ref):
        self.master = master_ref

    def call(self, *args, **kwargs):
        """
        Method that calls :meth:`forward` and returns parameter values of the
        guide as a `tuple` instead of a `dict`, which is a requirement for
        JIT tracing. Unlike :meth:`forward`, this method can be traced by
        :func:`torch.jit.trace_module`.

        .. warning::
            This method may be removed once PyTorch JIT tracer starts accepting
            `dict` as valid return types. See
            `issue <https://github.com/pytorch/pytorch/issues/27743>_`.
        """
        result = self(*args, **kwargs)
        return tuple(v for _, v in sorted(result.items()))

    def sample_latent(*args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        pass

    def __setattr__(self, name, value):
        if isinstance(value, AutoGuide):
            master_ref = self if self.master is None else self.master
            value._update_master(weakref.ref(master_ref))
        super().__setattr__(name, value)

    def _create_plates(self, *args, **kwargs):
        if self.master is None:
            if self.create_plates is None:
                self.plates = {}
            else:
                plates = self.create_plates(*args, **kwargs)
                if isinstance(plates, pyro.plate):
                    plates = [plates]
                assert all(
                    isinstance(p, pyro.plate) for p in plates
                ), "create_plates() returned a non-plate"
                self.plates = {p.name: p for p in plates}
            for name, frame in sorted(self._prototype_frames.items()):
                if name not in self.plates:
                    full_size = getattr(frame, "full_size", frame.size)
                    self.plates[name] = pyro.plate(
                        name, full_size, dim=frame.dim, subsample_size=frame.size
                    )
        else:
            assert (
                self.create_plates is None
            ), "Cannot pass create_plates() to non-master guide"
            self.plates = self.master().plates
        return self.plates

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        model = poutine.block(self.model, prototype_hide_fn)
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(
            *args, **kwargs
        )
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._prototype_frames = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._prototype_frames[frame.name] = frame
                else:
                    raise NotImplementedError(
                        "AutoGuide does not support sequential pyro.plate"
                    )

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        raise NotImplementedError


class AutoGuideList(AutoGuide, nn.ModuleList):
    """
    Container class to combine multiple automatic guides.

    Example usage::

        guide = AutoGuideList(my_model)
        guide.append(AutoDiagonalNormal(poutine.block(model, hide=["assignment"])))
        guide.append(AutoDiscreteParallel(poutine.block(model, expose=["assignment"])))
        svi = SVI(model, guide, optim, Trace_ELBO())

    :param callable model: a Pyro model
    """

    def _check_prototype(self, part_trace):
        for name, part_site in part_trace.nodes.items():
            if part_site["type"] != "sample":
                continue
            self_site = self.prototype_trace.nodes[name]
            assert part_site["fn"].batch_shape == self_site["fn"].batch_shape
            assert part_site["fn"].event_shape == self_site["fn"].event_shape
            assert part_site["value"].shape == self_site["value"].shape

    def _update_master(self, master_ref):
        self.master = master_ref
        for submodule in self:
            submodule._update_master(master_ref)

    def append(self, part):
        """
        Add an automatic guide for part of the model. The guide should
        have been created by blocking the model to restrict to a subset of
        sample sites. No two parts should operate on any one sample site.

        :param part: a partial guide to add
        :type part: AutoGuide or callable
        """
        if not isinstance(part, AutoGuide):
            part = AutoCallable(self.model, part)
        if part.master is not None:
            raise RuntimeError(
                "The module `{}` is already added.".format(self._pyro_name)
            )
        setattr(self, str(len(self)), part)

    def add(self, part):
        """Deprecated alias for :meth:`append`."""
        warnings.warn(
            "The method `.add` has been deprecated in favor of `.append`.",
            DeprecationWarning,
        )
        self.append(part)

    def forward(self, *args, **kwargs):
        """
        A composite guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        # create all plates
        self._create_plates(*args, **kwargs)

        # run slave guides
        result = {}
        for part in self:
            result.update(part(*args, **kwargs))
        return result

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        result = {}
        for part in self:
            result.update(part.median(*args, **kwargs))
        return result

    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns the posterior quantile values of each latent variable.

        :param list quantiles: A list of requested quantiles between 0 and 1.
        :returns: A dict mapping sample site name to quantiles tensor.
        :rtype: dict
        """
        result = {}
        for part in self:
            result.update(part.quantiles(quantiles, *args, **kwargs))
        return result


class AutoCallable(AutoGuide):
    """
    :class:`AutoGuide` wrapper for simple callable guides.

    This is used internally for composing autoguides with custom user-defined
    guides that are simple callables, e.g.::

        def my_local_guide(*args, **kwargs):
            ...

        guide = AutoGuideList(model)
        guide.add(AutoDelta(poutine.block(model, expose=['my_global_param']))
        guide.add(my_local_guide)  # automatically wrapped in an AutoCallable

    To specify a median callable, you can instead::

        def my_local_median(*args, **kwargs)
            ...

        guide.add(AutoCallable(model, my_local_guide, my_local_median))

    For more complex guides that need e.g. access to plates, users should
    instead subclass ``AutoGuide``.

    :param callable model: a Pyro model
    :param callable guide: a Pyro guide (typically over only part of the model)
    :param callable median: an optional callable returning a dict mapping
        sample site name to computed median tensor.
    """

    def __init__(self, model, guide, median=lambda *args, **kwargs: {}):
        super().__init__(model)
        self._guide = guide
        self.median = median

    def forward(self, *args, **kwargs):
        result = self._guide(*args, **kwargs)
        return {} if result is None else result


class AutoDelta(AutoGuide):
    """
    This implementation of :class:`AutoGuide` uses Delta distributions to
    construct a MAP guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    .. note:: This class does MAP inference in constrained space.

    Usage::

        guide = AutoDelta(model)
        svi = SVI(model, guide, ...)

    Latent variables are initialized using ``init_loc_fn()``. To change the
    default behavior, create a custom ``init_loc_fn()`` as described in
    :ref:`autoguide-initialization` , for example::

        def my_init_fn(site):
            if site["name"] == "level":
                return torch.tensor([-1., 0., 1.])
            if site["name"] == "concentration":
                return torch.ones(k)
            return init_to_sample(site)

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    def __init__(self, model, init_loc_fn=init_to_median, *, create_plates=None):
        self.init_loc_fn = init_loc_fn
        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        # Initialize guide params
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            value = site["value"].detach()
            event_dim = site["fn"].event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = getattr(frame, "full_size", frame.size)
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    value = periodic_repeat(value, full_size, dim).contiguous()

            value = PyroParam(value, site["fn"].support, event_dim)
            with helpful_support_errors(site):
                _deep_setattr(self, name, value)

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                attr_get = operator.attrgetter(name)
                result[name] = pyro.sample(
                    name, dist.Delta(attr_get(self), event_dim=site["fn"].event_dim)
                )
        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        result = self(*args, **kwargs)
        return {k: v.detach() for k, v in result.items()}


class AutoNormal(AutoGuide):
    """This implementation of :class:`AutoGuide` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    It should be equivalent to :class: `AutoDiagonalNormal` , but with
    more convenient site names and with better support for
    :class:`~pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO` .

    In :class:`AutoDiagonalNormal` , if your model has N named
    parameters with dimensions k_i and sum k_i = D, you get a single
    vector of length D for your mean, and a single vector of length D
    for sigmas.  This guide gives you N distinct normals that you can
    call by name.

    Usage::

        guide = AutoNormal(model)
        svi = SVI(model, guide, ...)

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    scale_constraint = constraints.softplus_positive

    def __init__(
        self, model, *, init_loc_fn=init_to_feasible, init_scale=0.1, create_plates=None
    ):
        self.init_loc_fn = init_loc_fn

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self.locs = PyroModule()
        self.scales = PyroModule()

        # Initialize guide params
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect unconstrained event_dims, which may differ from constrained event_dims.
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support).inv(site["value"].detach()).detach()
                )
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = getattr(frame, "full_size", frame.size)
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()
            init_scale = torch.full_like(init_loc, self._init_scale)

            _deep_setattr(
                self.locs, name, PyroParam(init_loc, constraints.real, event_dim)
            )
            _deep_setattr(
                self.scales,
                name,
                PyroParam(init_scale, self.scale_constraint, event_dim),
            )

    def _get_loc_and_scale(self, name):
        site_loc = _deep_getattr(self.locs, name)
        site_scale = _deep_getattr(self.scales, name)
        return site_loc, site_scale

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    dist.Normal(
                        site_loc,
                        site_scale,
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                value = transform(unconstrained_latent)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = transform.inv.log_abs_det_jacobian(
                        value,
                        unconstrained_latent,
                    )
                    log_density = sum_rightmost(
                        log_density,
                        log_density.dim() - value.dim() + site["fn"].event_dim,
                    )
                delta_dist = dist.Delta(
                    value,
                    log_density=log_density,
                    event_dim=site["fn"].event_dim,
                )

                result[name] = pyro.sample(name, delta_dist)

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, _ = self._get_loc_and_scale(name)
            median = biject_to(site["fn"].support)(site_loc)
            if median is site_loc:
                median = median.clone()
            medians[name] = median

        return medians

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a tensor of quantile values.
        :rtype: dict
        """
        results = {}

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, site_scale = self._get_loc_and_scale(name)

            site_quantiles = torch.tensor(
                quantiles, dtype=site_loc.dtype, device=site_loc.device
            )
            site_quantiles = site_quantiles.reshape((-1,) + (1,) * site_loc.dim())
            site_quantiles_values = dist.Normal(site_loc, site_scale).icdf(
                site_quantiles
            )
            constrained_site_quantiles = biject_to(site["fn"].support)(
                site_quantiles_values
            )
            results[name] = constrained_site_quantiles

        return results


class AutoContinuous(AutoGuide):
    """
    Base class for implementations of continuous-valued Automatic
    Differentiation Variational Inference [1].

    This uses :mod:`torch.distributions.transforms` to transform each
    constrained latent variable to an unconstrained space, then concatenate all
    variables into a single unconstrained latent variable.  Each derived class
    implements a :meth:`get_posterior` method returning a distribution over
    this single unconstrained latent variable.

    Assumes model structure and latent dimension are fixed, and all latent
    variables are continuous.

    :param callable model: a Pyro model

    Reference:

    [1] `Automatic Differentiation Variational Inference`,
        Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M.
        Blei

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """

    def __init__(self, model, init_loc_fn=init_to_median):
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self._unconstrained_shapes = {}
        self._cond_indep_stacks = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect the shapes of unconstrained values.
            # These may differ from the shapes of constrained values.
            with helpful_support_errors(site):
                self._unconstrained_shapes[name] = (
                    biject_to(site["fn"].support).inv(site["value"]).shape
                )

            # Collect independence contexts.
            self._cond_indep_stacks[name] = site["cond_indep_stack"]

        self.latent_dim = sum(
            _product(shape) for shape in self._unconstrained_shapes.values()
        )
        if self.latent_dim == 0:
            raise RuntimeError(
                "{} found no latent variables; Use an empty guide instead".format(
                    type(self).__name__
                )
            )

    def _init_loc(self):
        """
        Creates an initial latent vector using a per-site init function.
        """
        parts = []
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            constrained_value = site["value"].detach()
            unconstrained_value = biject_to(site["fn"].support).inv(constrained_value)
            parts.append(unconstrained_value.reshape(-1))
        latent = torch.cat(parts)
        assert latent.size() == (self.latent_dim,)
        return latent

    def get_base_dist(self):
        """
        Returns the base distribution of the posterior when reparameterized
        as a :class:`~pyro.distributions.TransformedDistribution`. This
        should not depend on the model's `*args, **kwargs`.

        .. code-block:: python

          posterior = TransformedDistribution(self.get_base_dist(), self.get_transform(*args, **kwargs))

        :return: :class:`~pyro.distributions.TorchDistribution` instance representing the base distribution.
        """
        raise NotImplementedError

    def get_transform(self, *args, **kwargs):
        """
        Returns the transform applied to the base distribution when the posterior
        is reparameterized as a :class:`~pyro.distributions.TransformedDistribution`.
        This may depend on the model's `*args, **kwargs`.

        .. code-block:: python

          posterior = TransformedDistribution(self.get_base_dist(), self.get_transform(*args, **kwargs))

        :return: a :class:`~torch.distributions.Transform` instance.
        """
        raise NotImplementedError

    def get_posterior(self, *args, **kwargs):
        """
        Returns the posterior distribution.
        """
        base_dist = self.get_base_dist()
        transform = self.get_transform(*args, **kwargs)
        return dist.TransformedDistribution(base_dist, transform)

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        pos_dist = self.get_posterior(*args, **kwargs)
        return pyro.sample(
            "_{}_latent".format(self._pyro_name), pos_dist, infer={"is_auxiliary": True}
        )

    def _unpack_latent(self, latent):
        """
        Unpacks a packed latent tensor, iterating over tuples of the form::

            (site, unconstrained_value)
        """
        batch_shape = latent.shape[
            :-1
        ]  # for plates outside of _setup_prototype, e.g. parallel particles
        pos = 0
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            constrained_shape = site["value"].shape
            unconstrained_shape = self._unconstrained_shapes[name]
            size = _product(unconstrained_shape)
            event_dim = (
                site["fn"].event_dim + len(unconstrained_shape) - len(constrained_shape)
            )
            unconstrained_shape = torch.broadcast_shapes(
                unconstrained_shape, batch_shape + (1,) * event_dim
            )
            unconstrained_value = latent[..., pos : pos + size].view(
                unconstrained_shape
            )
            yield site, unconstrained_value
            pos += size
        if not torch._C._get_tracing_state():
            assert pos == latent.size(-1)

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        latent = self.sample_latent(*args, **kwargs)
        plates = self._create_plates(*args, **kwargs)

        # unpack continuous latent samples
        result = {}
        for site, unconstrained_value in self._unpack_latent(latent):
            name = site["name"]
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)
            if poutine.get_mask() is False:
                log_density = 0.0
            else:
                log_density = transform.inv.log_abs_det_jacobian(
                    value,
                    unconstrained_value,
                )
                log_density = sum_rightmost(
                    log_density,
                    log_density.dim() - value.dim() + site["fn"].event_dim,
                )
            delta_dist = dist.Delta(
                value,
                log_density=log_density,
                event_dim=site["fn"].event_dim,
            )

            with ExitStack() as stack:
                for frame in self._cond_indep_stacks[name]:
                    stack.enter_context(plates[frame.name])
                result[name] = pyro.sample(name, delta_dist)

        return result

    def _loc_scale(self, *args, **kwargs):
        """
        :returns: a tuple ``(loc, scale)`` used by :meth:`median` and
            :meth:`quantiles`
        """
        raise NotImplementedError

    @torch.no_grad()
    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        loc, _ = self._loc_scale(*args, **kwargs)
        loc = loc.detach()
        return {
            site["name"]: biject_to(site["fn"].support)(unconstrained_value)
            for site, unconstrained_value in self._unpack_latent(loc)
        }

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a tensor of quantile values.
        :rtype: dict
        """
        loc, scale = self._loc_scale(*args, **kwargs)
        quantiles = torch.tensor(
            quantiles, dtype=loc.dtype, device=loc.device
        ).unsqueeze(-1)
        latents = dist.Normal(loc, scale).icdf(quantiles)
        result = {}
        for latent in latents:
            for site, unconstrained_value in self._unpack_latent(latent):
                result.setdefault(site["name"], []).append(
                    biject_to(site["fn"].support)(unconstrained_value)
                )
        result = {k: torch.stack(v) for k, v in result.items()}
        return result


class AutoMultivariateNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Cholesky
    factorization of a Multivariate Normal distribution to construct a guide
    over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.

    Usage::

        guide = AutoMultivariateNormal(model)
        svi = SVI(model, guide, ...)

    By default the mean vector is initialized by ``init_loc_fn()`` and the
    Cholesky factor is initialized to the identity times a small factor.

    :param callable model: A generative model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    """

    scale_tril_constraint = constraints.softplus_lower_cholesky

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        self.scale_tril = PyroParam(
            eye_like(self.loc, self.latent_dim) * self._init_scale,
            self.scale_tril_constraint,
        )

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.zeros_like(self.loc)
        ).to_event(1)

    def get_transform(self, *args, **kwargs):
        return dist.transforms.LowerCholeskyAffine(self.loc, scale_tril=self.scale_tril)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a MultivariateNormal posterior distribution.
        """
        return dist.MultivariateNormal(self.loc, scale_tril=self.scale_tril)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale_tril.diag()


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(model)
        svi = SVI(model, guide, ...)

    By default the mean vector is initialized to zero and the scale is
    initialized to the identity times a small factor.

    :param callable model: A generative model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    """

    scale_constraint = constraints.softplus_positive

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            self.loc.new_full((self.latent_dim,), self._init_scale),
            self.scale_constraint,
        )

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.zeros_like(self.loc)
        ).to_event(1)

    def get_transform(self, *args, **kwargs):
        return dist.transforms.AffineTransform(self.loc, self.scale)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Normal posterior distribution.
        """
        return dist.Normal(self.loc, self.scale).to_event(1)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale


class AutoLowRankMultivariateNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a low rank plus
    diagonal Multivariate Normal distribution to construct a guide
    over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.

    Usage::

        guide = AutoLowRankMultivariateNormal(model, rank=10)
        svi = SVI(model, guide, ...)

    By default the ``cov_diag`` is initialized to a small constant and the
    ``cov_factor`` is initialized randomly such that on average
    ``cov_factor.matmul(cov_factor.t())`` has the same scale as ``cov_diag``.

    :param callable model: A generative model.
    :param rank: The rank of the low-rank part of the covariance matrix.
        Defaults to approximately ``sqrt(latent dim)``.
    :type rank: int or None
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Approximate initial scale for the standard
        deviation of each (unconstrained transformed) latent variable.
    """

    scale_constraint = constraints.softplus_positive

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1, rank=None):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        if not (rank is None or isinstance(rank, int) and rank > 0):
            raise ValueError("Expected rank > 0 but got {}".format(rank))
        self._init_scale = init_scale
        self.rank = rank
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        if self.rank is None:
            self.rank = int(round(self.latent_dim ** 0.5))
        self.scale = PyroParam(
            self.loc.new_full((self.latent_dim,), 0.5 ** 0.5 * self._init_scale),
            constraint=self.scale_constraint,
        )
        self.cov_factor = nn.Parameter(
            self.loc.new_empty(self.latent_dim, self.rank).normal_(
                0, 1 / self.rank ** 0.5
            )
        )

    def get_posterior(self, *args, **kwargs):
        """
        Returns a LowRankMultivariateNormal posterior distribution.
        """
        scale = self.scale
        cov_factor = self.cov_factor * scale.unsqueeze(-1)
        cov_diag = scale * scale
        return dist.LowRankMultivariateNormal(self.loc, cov_factor, cov_diag)

    def _loc_scale(self, *args, **kwargs):
        scale = self.scale * (self.cov_factor.pow(2).sum(-1) + 1).sqrt()
        return self.loc, scale


class AutoNormalizingFlow(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a sequence of bijective transforms
    (e.g. various :mod:`~pyro.distributions.TransformModule` subclasses)
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    Usage::

        transform_init = partial(iterated, block_autoregressive,
                                 repeats=2)
        guide = AutoNormalizingFlow(model, transform_init)
        svi = SVI(model, guide, ...)

    :param callable model: a generative model
    :param init_transform_fn: a callable which when provided with the latent
        dimension returns an instance of :class:`~torch.distributions.Transform`
        , or :class:`~pyro.distributions.TransformModule` if the transform has
        trainable params.
    """

    def __init__(self, model, init_transform_fn):
        super().__init__(model, init_loc_fn=init_to_feasible)
        self._init_transform_fn = init_transform_fn
        self.transform = None
        self._prototype_tensor = torch.tensor(0.0)

    def get_base_dist(self):
        loc = self._prototype_tensor.new_zeros(1)
        scale = self._prototype_tensor.new_ones(1)
        return dist.Normal(loc, scale).expand([self.latent_dim]).to_event(1)

    def get_transform(self, *args, **kwargs):
        return self.transform

    def get_posterior(self, *args, **kwargs):
        if self.transform is None:
            self.transform = self._init_transform_fn(self.latent_dim)
            # Update prototype tensor in case transform parameters
            # device/dtype is not the same as default tensor type.
            for _, p in self.named_pyro_params():
                self._prototype_tensor = p
                break
        return super().get_posterior(*args, **kwargs)


class AutoIAFNormal(AutoNormalizingFlow):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a :class:`~pyro.distributions.transforms.AffineAutoregressive`
    to construct a guide over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.

    Usage::

        guide = AutoIAFNormal(model, hidden_dim=latent_dim)
        svi = SVI(model, guide, ...)

    :param callable model: a generative model
    :param list[int] hidden_dim: number of hidden dimensions in the IAF
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.

        .. warning::

            This argument is only to preserve backwards compatibility
            and has no effect in practice.

    :param int num_transforms: number of :class:`~pyro.distributions.transforms.AffineAutoregressive`
        transforms to use in sequence.
    :param init_transform_kwargs: other keyword arguments taken by
        :func:`~pyro.distributions.transforms.affine_autoregressive`.
    """

    def __init__(
        self,
        model,
        hidden_dim=None,
        init_loc_fn=None,
        num_transforms=1,
        **init_transform_kwargs,
    ):
        if init_loc_fn:
            warnings.warn(
                "The `init_loc_fn` argument to AutoIAFNormal is not used in practice. "
                "Please consider removing, as this may be removed in a future release.",
                category=FutureWarning,
            )
        super().__init__(
            model,
            init_transform_fn=functools.partial(
                iterated,
                num_transforms,
                affine_autoregressive,
                hidden_dims=hidden_dim,
                **init_transform_kwargs,
            ),
        )


class AutoLaplaceApproximation(AutoContinuous):
    r"""
    Laplace approximation (quadratic approximation) approximates the posterior
    :math:`\log p(z | x)` by a multivariate normal distribution in the
    unconstrained space. Under the hood, it uses Delta distributions to
    construct a MAP guide over the entire (unconstrained) latent space. Its
    covariance is given by the inverse of the hessian of :math:`-\log p(x, z)`
    at the MAP point of `z`.

    Usage::

        delta_guide = AutoLaplaceApproximation(model)
        svi = SVI(model, delta_guide, ...)
        # ...then train the delta_guide...
        guide = delta_guide.laplace_approximation()

    By default the mean vector is initialized to an empirical prior median.

    :param callable model: a generative model
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())

    def get_posterior(self, *args, **kwargs):
        """
        Returns a Delta posterior distribution for MAP inference.
        """
        return dist.Delta(self.loc).to_event(1)

    def laplace_approximation(self, *args, **kwargs):
        """
        Returns a :class:`AutoMultivariateNormal` instance whose posterior's `loc` and
        `scale_tril` are given by Laplace approximation.
        """
        guide_trace = poutine.trace(self).get_trace(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(self.model, trace=guide_trace)
        ).get_trace(*args, **kwargs)
        loss = guide_trace.log_prob_sum() - model_trace.log_prob_sum()

        H = hessian(loss, self.loc)
        cov = H.inverse()
        loc = self.loc
        scale_tril = torch.linalg.cholesky(cov)

        gaussian_guide = AutoMultivariateNormal(self.model)
        gaussian_guide._setup_prototype(*args, **kwargs)
        # Set loc, scale_tril parameters as computed above.
        gaussian_guide.loc = loc
        gaussian_guide.scale_tril = scale_tril
        return gaussian_guide


class AutoDiscreteParallel(AutoGuide):
    """
    A discrete mean-field guide that learns a latent discrete distribution for
    each discrete site in the model.
    """

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        model = poutine.block(config_enumerate(self.model), prototype_hide_fn)
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(
            *args, **kwargs
        )
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._discrete_sites = []
        self._cond_indep_stacks = {}
        self._prototype_frames = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            if site["infer"].get("enumerate") != "parallel":
                raise NotImplementedError(
                    'Expected sample site "{}" to be discrete and '
                    "configured for parallel enumeration".format(name)
                )

            # collect discrete sample sites
            fn = site["fn"]
            Dist = type(fn)
            if Dist in (dist.Bernoulli, dist.Categorical, dist.OneHotCategorical):
                params = [
                    ("probs", fn.probs.detach().clone(), fn.arg_constraints["probs"])
                ]
            else:
                raise NotImplementedError("{} is not supported".format(Dist.__name__))
            self._discrete_sites.append((site, Dist, params))

            # collect independence contexts
            self._cond_indep_stacks[name] = site["cond_indep_stack"]
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._prototype_frames[frame.name] = frame
                else:
                    raise NotImplementedError(
                        "AutoDiscreteParallel does not support sequential pyro.plate"
                    )
        # Initialize guide params
        for site, Dist, param_spec in self._discrete_sites:
            name = site["name"]
            for param_name, param_init, param_constraint in param_spec:
                _deep_setattr(
                    self,
                    "{}_{}".format(name, param_name),
                    PyroParam(param_init, constraint=param_constraint),
                )

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)

        # enumerate discrete latent samples
        result = {}
        for site, Dist, param_spec in self._discrete_sites:
            name = site["name"]
            dist_params = {
                param_name: operator.attrgetter("{}_{}".format(name, param_name))(self)
                for param_name, param_init, param_constraint in param_spec
            }
            discrete_dist = Dist(**dist_params)

            with ExitStack() as stack:
                for frame in self._cond_indep_stacks[name]:
                    stack.enter_context(plates[frame.name])
                result[name] = pyro.sample(
                    name, discrete_dist, infer={"enumerate": "parallel"}
                )

        return result


def _config_auxiliary(msg):
    return {"is_auxiliary": True}


class AutoStructured(AutoGuide):
    """
    Structured guide whose conditional distributions are Delta, Normal,
    MultivariateNormal, or by a callable, and whose latent variables can depend
    on each other either linearly (in unconstrained space) or via shearing by a
    callable.

    Usage::

        def model(data):
            x = pyro.sample("x", dist.LogNormal(0, 1))
            with pyro.plate("plate", len(data)):
                y = pyro.sample("y", dist.Normal(0, 1))
                pyro.sample("z", dist.Normal(y, x), obs=data)

        guide = AutoStructured(
            model=model,
            conditionals={"x": "normal", "y": "normal"},
            dependencies={"x": {"y": "linear"}},
        )

    Once trained, this guide can be used with
    :class:`~pyro.infer.reparam.structured.StructuredReparam` to precondition a
    model for use in HMC and NUTS inference.

    .. note:: If you declare a dependency of a high-dimensional downstream
        variable on a low-dimensional upstream variable, you may want to use
        a lower learning rate for that weight, e.g.::

            def optim_config(param_name):
                config = {"lr": 0.01}
                if "deps.my_downstream.my_upstream" in param_name:
                    config["lr"] *= 0.1
                return config

            adam = pyro.optim.Adam(optim_config)

    :param callable model: A Pyro model.
    :param conditionals: Family of distribution with which to model each latent
        variable's conditional posterior. This should be a dict mapping each
        latent variable name to either a string in ("delta", "normal", or
        "mvn") or to a callable that returns a sample from a zero mean (or
        approximately centered) noise distribution (such callables typically
        call ``pyro.param()`` and ``pyro.sample()`` internally).
    :param dependencies: Dict mapping each site name to a dict of its upstream
        dependencies; each inner dict maps upstream site name to either the
        string "linear" or a callable that maps a *flattened* upstream
        perturbation to *flattened* downstream perturbation. The string
        "linear" is equivalent to ``nn.Linear(upstream.numel(),
        downstream.numel(), bias=False)``.  Dependencies must not contain
        cycles or self-loops.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    scale_constraint = constraints.softplus_positive
    scale_tril_constraint = constraints.softplus_lower_cholesky

    def __init__(
        self,
        model,
        *,
        conditionals: Dict[str, Union[str, Callable]] = "normal",
        dependencies: Dict[str, Dict[str, Union[str, Callable]]] = "linear",
        init_loc_fn=init_to_feasible,
        init_scale=0.1,
        create_plates=None,
    ):
        assert isinstance(conditionals, dict)
        for name, fn in conditionals.items():
            assert isinstance(name, str)
            assert isinstance(fn, str) or callable(fn)
        assert isinstance(dependencies, dict)
        for downstream, deps in dependencies.items():
            assert downstream in conditionals
            assert isinstance(deps, dict)
            for upstream, dep in deps.items():
                assert upstream in conditionals
                assert upstream != downstream
                assert isinstance(dep, str) or callable(dep)
        self.conditionals = conditionals
        self.dependencies = dependencies

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(f"Expected init_scale > 0. but got {init_scale}")
        self._init_scale = init_scale
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self.locs = PyroModule()
        self.scales = PyroModule()
        self.scale_trils = PyroModule()
        self.conds = PyroModule()
        self.deps = PyroModule()
        self._batch_shapes = {}
        self._unconstrained_event_shapes = {}

        # Collect unconstrained shapes.
        init_locs = {}
        numel = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support).inv(site["value"].detach()).detach()
                )
            self._batch_shapes[name] = site["fn"].batch_shape
            self._unconstrained_event_shapes[name] = init_loc.shape[
                len(site["fn"].batch_shape) :
            ]
            numel[name] = init_loc.numel()
            init_locs[name] = init_loc.reshape(-1)

        # Initialize guide params.
        children = defaultdict(list)
        num_pending = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Initialize location parameters.
            init_loc = init_locs[name]
            _deep_setattr(self.locs, name, PyroParam(init_loc))

            # Initialize parameters of conditional distributions.
            conditional = self.conditionals[name]
            if callable(conditional):
                _deep_setattr(self.conds, name, conditional)
            else:
                if conditional not in ("delta", "normal", "mvn"):
                    raise ValueError(f"Unsupported conditional type: {conditional}")
                if conditional in ("normal", "mvn"):
                    init_scale = torch.full_like(init_loc, self._init_scale)
                    _deep_setattr(
                        self.scales, name, PyroParam(init_scale, self.scale_constraint)
                    )
                if conditional == "mvn":
                    init_scale_tril = eye_like(init_loc, init_loc.numel())
                    _deep_setattr(
                        self.scale_trils,
                        name,
                        PyroParam(init_scale_tril, self.scale_tril_constraint),
                    )

            # Initialize dependencies on upstream variables.
            num_pending[name] = 0
            deps = PyroModule()
            _deep_setattr(self.deps, name, deps)
            for upstream, dep in self.dependencies.get(name, {}).items():
                assert upstream in self.prototype_trace.nodes
                children[upstream].append(name)
                num_pending[name] += 1
                if isinstance(dep, str) and dep == "linear":
                    dep = torch.nn.Linear(numel[upstream], numel[name], bias=False)
                    dep.weight.data.zero_()
                elif not callable(dep):
                    raise ValueError(
                        f"Expected either the string 'linear' or a callable, but got {dep}"
                    )
                _deep_setattr(deps, upstream, dep)

        # Topologically sort sites.
        self._sorted_sites = []
        while num_pending:
            name, count = min(num_pending.items(), key=lambda kv: (kv[1], kv[0]))
            assert count == 0, f"cyclic dependency: {name}"
            del num_pending[name]
            for child in children[name]:
                num_pending[child] -= 1
            site = self._compress_site(self.prototype_trace.nodes[name])
            self._sorted_sites.append((name, site))

        # Prune non-essential parts of the trace to save memory.
        for name, site in self.prototype_trace.nodes.items():
            site.clear()

    @staticmethod
    def _compress_site(site):
        # Save memory by retaining only necessary parts of the site.
        return {
            "name": site["name"],
            "type": site["type"],
            "cond_indep_stack": site["cond_indep_stack"],
            "fn": SimpleNamespace(
                support=site["fn"].support,
                event_dim=site["fn"].event_dim,
            ),
        }

    @poutine.infer_config(config_fn=_config_auxiliary)
    def get_deltas(self, save_params=None):
        deltas = {}
        aux_values = {}
        compute_density = poutine.get_mask() is not False
        for name, site in self._sorted_sites:
            if save_params is not None and name not in save_params:
                continue

            # Sample zero-mean blockwise independent Delta/Normal/MVN.
            log_density = 0.0
            loc = _deep_getattr(self.locs, name)
            zero = torch.zeros_like(loc)
            conditional = self.conditionals[name]
            if callable(conditional):
                aux_value = _deep_getattr(self.conds, name)()
            elif conditional == "delta":
                aux_value = zero
            elif conditional == "normal":
                aux_value = pyro.sample(
                    name + "_aux",
                    dist.Normal(zero, 1).to_event(1),
                    infer={"is_auxiliary": True},
                )
                scale = _deep_getattr(self.scales, name)
                aux_value = aux_value * scale
                if compute_density:
                    log_density = (-scale.log()).expand_as(aux_value)
            elif conditional == "mvn":
                # This overparametrizes by learning (scale,scale_tril),
                # enabling faster learning of the more-global scale parameter.
                aux_value = pyro.sample(
                    name + "_aux",
                    dist.Normal(zero, 1).to_event(1),
                    infer={"is_auxiliary": True},
                )
                scale = _deep_getattr(self.scales, name)
                scale_tril = _deep_getattr(self.scale_trils, name)
                aux_value = aux_value @ scale_tril.T * scale
                if compute_density:
                    log_density = (
                        -scale_tril.diagonal(dim1=-2, dim2=-1).log() - scale.log()
                    ).expand_as(aux_value)
            else:
                raise ValueError(f"Unsupported conditional type: {conditional}")

            # Accumulate upstream dependencies.
            # Note: by accumulating upstream dependencies before updating the
            # aux_values dict, we encode a block-sparse structure of the
            # precision matrix; if we had instead accumulated after updating
            # aux_values, we would encode a block-sparse structure of the
            # covariance matrix.
            # Note: these shear transforms have no effect on the Jacobian
            # determinant, and can therefore be excluded from the log_density
            # computation below, even for nonlinear dep().
            deps = _deep_getattr(self.deps, name)
            for upstream in self.dependencies.get(name, {}):
                dep = _deep_getattr(deps, upstream)
                aux_value = aux_value + dep(aux_values[upstream])
            aux_values[name] = aux_value

            # Shift by loc and reshape.
            batch_shape = torch.broadcast_shapes(
                aux_value.shape[:-1], self._batch_shapes[name]
            )
            unconstrained = (aux_value + loc).reshape(
                batch_shape + self._unconstrained_event_shapes[name]
            )
            if not is_identically_zero(log_density):
                log_density = log_density.reshape(batch_shape + (-1,)).sum(-1)

            # Transform to constrained space.
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained)
            if compute_density and conditional != "delta":
                assert transform.codomain.event_dim == site["fn"].event_dim
                log_density = log_density + transform.inv.log_abs_det_jacobian(
                    value, unconstrained
                )

            # Create a reparametrized Delta distribution.
            deltas[name] = dist.Delta(value, log_density, site["fn"].event_dim)

        return deltas

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        deltas = self.get_deltas()
        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self._sorted_sites:
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                result[name] = pyro.sample(name, deltas[name])

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        result = {}
        for name, site in self._sorted_sites:
            loc = _deep_getattr(self.locs, name).detach()
            shape = self._batch_shapes[name] + self._unconstrained_event_shapes[name]
            loc = loc.reshape(shape)
            result[name] = biject_to(site["fn"].support)(loc)
        return result
