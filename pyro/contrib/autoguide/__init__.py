"""
The :mod:`pyro.contrib.autoguide` module provides algorithms to automatically
generate guides from simple models, for use in :class:`~pyro.infer.svi.SVI`.
For example to generate a mean field Gaussian guide::

    def model():
        ...

    guide = AutoDiagonalNormal(model)  # a mean field guide
    svi = SVI(model, guide, Adam({'lr': 1e-3}), Trace_ELBO())

Automatic guides can also be combined using :func:`pyro.poutine.block` and
:class:`AutoGuideList`.
"""

from __future__ import absolute_import, division, print_function

import numbers
import weakref

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.autoguide.initialization import (InitMessenger, init_to_feasible, init_to_mean, init_to_median,
                                                   init_to_sample)
from pyro.contrib.util import hessian
from pyro.distributions.util import broadcast_shape, eye_like, sum_rightmost
from pyro.infer.enum import config_enumerate
from pyro.nn import AutoRegressiveNN
from pyro.poutine.util import prune_subsample_sites

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2

__all__ = [
    'AutoCallable',
    'AutoContinuous',
    'AutoDelta',
    'AutoDiagonalNormal',
    'AutoDiscreteParallel',
    'AutoGuide',
    'AutoGuideList',
    'AutoIAFNormal',
    'AutoLaplaceApproximation',
    'AutoLowRankMultivariateNormal',
    'AutoMultivariateNormal',
    'init_to_feasible',
    'init_to_mean',
    'init_to_median',
    'init_to_sample',
    'mean_field_entropy',
]


def _product(shape):
    """
    Computes the product of the dimensions of a given shape tensor
    """
    result = 1
    for size in shape:
        result *= size
    return result


class AutoGuide(object):
    """
    Base class for automatic guides.

    Derived classes must implement the :meth:`__call__` method.

    Auto guides can be used individually or combined in an
    :class:`AutoGuideList` object.

    :param callable model: a pyro model
    :param str prefix: a prefix that will be prefixed to all param internal sites
    """

    def __init__(self, model, prefix="auto"):
        self.master = None
        self.model = model
        self.prefix = prefix
        self.prototype_trace = None
        self._plates = {}

    def __call__(self, *args, **kwargs):
        """
        A guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        raise NotImplementedError

    def sample_latent(*args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        pass

    def _create_plates(self):
        if self.master is not None:
            return self.master().plates
        return {frame.name: pyro.plate(frame.name, frame.size, dim=frame.dim)
                for frame in sorted(self._plates.values())}

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        self.prototype_trace = poutine.block(poutine.trace(self.model).get_trace)(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._plates = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._plates[frame.name] = frame
                else:
                    raise NotImplementedError("AutoGuideList does not support sequential pyro.plate")

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        raise NotImplementedError


class AutoGuideList(AutoGuide):
    """
    Container class to combine multiple automatic guides.

    Example usage::

        guide = AutoGuideList(my_model)
        guide.add(AutoDiagonalNormal(poutine.block(model, hide=["assignment"])))
        guide.add(AutoDiscreteParallel(poutine.block(model, expose=["assignment"])))
        svi = SVI(model, guide, optim, Trace_ELBO())

    :param callable model: a Pyro model
    :param str prefix: a prefix that will be prefixed to all param internal sites
    """

    def __init__(self, model, prefix="auto"):
        super(AutoGuideList, self).__init__(model, prefix)
        self.parts = []
        self.plates = {}

    def _check_prototype(self, part_trace):
        for name, part_site in part_trace.nodes.items():
            if part_site["type"] != "sample":
                continue
            self_site = self.prototype_trace.nodes[name]
            assert part_site["fn"].batch_shape == self_site["fn"].batch_shape
            assert part_site["fn"].event_shape == self_site["fn"].event_shape
            assert part_site["value"].shape == self_site["value"].shape

    def add(self, part):
        """
        Add an automatic guide for part of the model. The guide should
        have been created by blocking the model to restrict to a subset of
        sample sites. No two parts should operate on any one sample site.

        :param part: a partial guide to add
        :type part: AutoGuide or callable
        """
        if not isinstance(part, AutoGuide):
            part = AutoCallable(self.model, part)
        self.parts.append(part)
        assert part.master is None
        part.master = weakref.ref(self)

    def __call__(self, *args, **kwargs):
        """
        A composite guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        # create all plates
        self.plates = {frame.name: pyro.plate(frame.name, frame.size, dim=frame.dim)
                       for frame in sorted(self._plates.values())}

        # run slave guides
        result = {}
        for part in self.parts:
            result.update(part(*args, **kwargs))
        return result

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        result = {}
        for part in self.parts:
            result.update(part.median(*args, **kwargs))
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
        super(AutoCallable, self).__init__(model, prefix="")
        self._guide = guide
        self.median = median

    def __call__(self, *args, **kwargs):
        result = self._guide(*args, **kwargs)
        return {} if result is None else result


class AutoDelta(AutoGuide):
    """
    This implementation of :class:`AutoGuide` uses Delta distributions to
    construct a MAP guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    ..note:: This class does MAP inference in constrained space.

    Usage::

        guide = AutoDelta(model)
        svi = SVI(model, guide, ...)

    By default latent variables are initialized using ``init_loc_fn()``. To
    change this default behavior the user should call :func:`pyro.param` before
    beginning inference, with ``"auto_"`` prefixed to the targeted sample site
    names e.g. for sample sites named "level" and "concentration", initialize
    via::

        pyro.param("auto_level", torch.tensor([-1., 0., 1.]))
        pyro.param("auto_concentration", torch.ones(k),
                   constraint=constraints.positive)

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """
    def __init__(self, model, prefix="auto", init_loc_fn=init_to_median):
        self.init_loc_fn = init_loc_fn
        model = InitMessenger(self._init_loc_fn)(model)
        super(AutoDelta, self).__init__(model, prefix=prefix)

    def _init_loc_fn(self, site):
        name = "{}_{}".format(self.prefix, site["name"])
        store = pyro.get_param_store()
        if name in store:
            return store[name]
        return self.init_loc_fn(site)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                value = pyro.param("{}_{}".format(self.prefix, name),
                                   site["value"].detach(),
                                   constraint=site["fn"].support)
                result[name] = pyro.sample(name, dist.Delta(value, event_dim=site["fn"].event_dim))
        return result

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        return self(*args, **kwargs)


class AutoContinuous(AutoGuide):
    """
    Base class for implementations of continuous-valued Automatic
    Differentiation Variational Inference [1].

    Each derived class implements its own :meth:`get_posterior` method.

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
    def __init__(self, model, prefix="auto", init_loc_fn=init_to_median):
        model = InitMessenger(init_loc_fn)(model)
        super(AutoContinuous, self).__init__(model, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(AutoContinuous, self)._setup_prototype(*args, **kwargs)
        self._unconstrained_shapes = {}
        self._cond_indep_stacks = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect the shapes of unconstrained values.
            # These may differ from the shapes of constrained values.
            self._unconstrained_shapes[name] = biject_to(site["fn"].support).inv(site["value"]).shape

            # Collect independence contexts.
            self._cond_indep_stacks[name] = site["cond_indep_stack"]

        self.latent_dim = sum(_product(shape) for shape in self._unconstrained_shapes.values())
        if self.latent_dim == 0:
            raise RuntimeError('{} found no latent variables; Use an empty guide instead'.format(type(self).__name__))

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

    def get_posterior(self, *args, **kwargs):
        """
        Returns the posterior distribution.
        """
        raise NotImplementedError

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        pos_dist = self.get_posterior(*args, **kwargs)
        return pyro.sample("_{}_latent".format(self.prefix), pos_dist, infer={"is_auxiliary": True})

    def _unpack_latent(self, latent):
        """
        Unpacks a packed latent tensor, iterating over tuples of the form::

            (site, unconstrained_value)
        """
        batch_shape = latent.shape[:-1]  # for plates outside of _setup_prototype, e.g. parallel particles
        pos = 0
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            constrained_shape = site["value"].shape
            unconstrained_shape = self._unconstrained_shapes[name]
            size = _product(unconstrained_shape)
            event_dim = site["fn"].event_dim + len(unconstrained_shape) - len(constrained_shape)
            unconstrained_shape = broadcast_shape(unconstrained_shape,
                                                  batch_shape + (1,) * event_dim)
            unconstrained_value = latent[..., pos:pos + size].view(unconstrained_shape)
            yield site, unconstrained_value
            pos += size
        if not torch._C._get_tracing_state():
            assert pos == latent.size(-1)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        latent = self.sample_latent(*args, **kwargs)
        plates = self._create_plates()

        # unpack continuous latent samples
        result = {}
        for site, unconstrained_value in self._unpack_latent(latent):
            name = site["name"]
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)
            log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
            log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + site["fn"].event_dim)
            delta_dist = dist.Delta(value, log_density=log_density, event_dim=site["fn"].event_dim)

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

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        loc, _ = self._loc_scale(*args, **kwargs)
        return {site["name"]: biject_to(site["fn"].support)(unconstrained_value)
                for site, unconstrained_value in self._unpack_latent(loc)}

    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        loc, scale = self._loc_scale(*args, **kwargs)
        quantiles = torch.tensor(quantiles, dtype=loc.dtype, device=loc.device).unsqueeze(-1)
        latents = dist.Normal(loc, scale).icdf(quantiles)
        result = {}
        for latent in latents:
            for site, unconstrained_value in self._unpack_latent(latent):
                result.setdefault(site["name"], []).append(biject_to(site["fn"].support)(unconstrained_value))
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

    By default the mean vector is initialized to zero and the Cholesky factor
    is initialized to the identity.  To change this default behavior the user
    should call :func:`pyro.param` before beginning inference, e.g.::

        latent_dim = 10
        pyro.param("auto_loc", torch.randn(latent_dim))
        pyro.param("auto_scale_tril", torch.tril(torch.rand(latent_dim)),
                   constraint=constraints.lower_cholesky)
    """

    def get_posterior(self, *args, **kwargs):
        """
        Returns a MultivariateNormal posterior distribution.
        """
        loc = pyro.param("{}_loc".format(self.prefix), self._init_loc)
        scale_tril = pyro.param("{}_scale_tril".format(self.prefix),
                                lambda: eye_like(loc, self.latent_dim),
                                constraint=constraints.lower_cholesky)
        return dist.MultivariateNormal(loc, scale_tril=scale_tril)

    def _loc_scale(self, *args, **kwargs):
        loc = pyro.param("{}_loc".format(self.prefix))
        scale = pyro.param("{}_scale_tril".format(self.prefix)).diag()
        return loc, scale


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(model)
        svi = SVI(model, guide, ...)

    By default the mean vector is initialized to zero and the scale is
    initialized to the identity.  To change this default behavior the user
    should call :func:`pyro.param` before beginning inference, e.g.::

        latent_dim = 10
        pyro.param("auto_loc", torch.randn(latent_dim))
        pyro.param("auto_scale", torch.ones(latent_dim),
                   constraint=constraints.positive)
    """

    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Normal posterior distribution.
        """
        loc = pyro.param("{}_loc".format(self.prefix), self._init_loc)
        scale = pyro.param("{}_scale".format(self.prefix),
                           lambda: loc.new_ones(self.latent_dim),
                           constraint=constraints.positive)
        return dist.Normal(loc, scale).to_event(1)

    def _loc_scale(self, *args, **kwargs):
        loc = pyro.param("{}_loc".format(self.prefix))
        scale = pyro.param("{}_scale".format(self.prefix))
        return loc, scale


class AutoLowRankMultivariateNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a low rank plus
    diagonal Multivariate Normal distribution to construct a guide
    over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.

    Usage::

        guide = AutoLowRankMultivariateNormal(model, rank=10)
        svi = SVI(model, guide, ...)

    By default the ``cov_diag`` is initialized to 1/2 and the ``cov_factor`` is
    intialized randomly such that ``cov_factor.matmul(cov_factor.t())`` is half the
    identity matrix. To change this default behavior the user
    should call :func:`pyro.param` before beginning inference, e.g.::

        latent_dim = 10
        pyro.param("auto_loc", torch.randn(latent_dim))
        pyro.param("auto_cov_factor", torch.randn(latent_dim, rank)))
        pyro.param("auto_cov_diag", torch.randn(latent_dim).exp()),
                   constraint=constraints.positive)

    :param callable model: a generative model
    :param int rank: the rank of the low-rank part of the covariance matrix
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param str prefix: a prefix that will be prefixed to all param internal sites
    """

    def __init__(self, model, prefix="auto", init_loc_fn=init_to_median, rank=1):
        if not isinstance(rank, numbers.Number) or not rank > 0:
            raise ValueError("Expected rank > 0 but got {}".format(rank))
        self.rank = rank
        super(AutoLowRankMultivariateNormal, self).__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a LowRankMultivariateNormal posterior distribution.
        """
        loc = pyro.param("{}_loc".format(self.prefix), self._init_loc)
        factor = pyro.param("{}_cov_factor".format(self.prefix),
                            lambda: loc.new_empty(self.latent_dim, self.rank).normal_(0, (0.5 / self.rank) ** 0.5))
        diagonal = pyro.param("{}_cov_diag".format(self.prefix),
                              lambda: loc.new_full((self.latent_dim,), 0.5),
                              constraint=constraints.positive)
        return dist.LowRankMultivariateNormal(loc, factor, diagonal)

    def _loc_scale(self, *args, **kwargs):
        loc = pyro.param("{}_loc".format(self.prefix))
        factor = pyro.param("{}_cov_factor".format(self.prefix))
        diagonal = pyro.param("{}_cov_diag".format(self.prefix))
        scale = (factor.pow(2).sum(-1) + diagonal).sqrt()
        return loc, scale


class AutoIAFNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a :class:`~pyro.distributions.iaf.InverseAutoregressiveFlow`
    to construct a guide over the entire latent space. The guide does not depend on the model's
    ``*args, **kwargs``.

    Usage::

        guide = AutoIAFNormal(model, hidden_dim=latent_dim)
        svi = SVI(model, guide, ...)

    :param callable model: a generative model
    :param int hidden_dim: number of hidden dimensions in the IAF
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param str prefix: a prefix that will be prefixed to all param internal sites
    """

    def __init__(self, model, hidden_dim=None, prefix="auto", init_loc_fn=init_to_median):
        self.hidden_dim = hidden_dim
        self.arn = None
        super(AutoIAFNormal, self).__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Normal posterior distribution transformed by
        :class:`~pyro.distributions.iaf.InverseAutoregressiveFlow`.
        """
        if self.latent_dim == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        if self.hidden_dim is None:
            self.hidden_dim = self.latent_dim
        if self.arn is None:
            self.arn = AutoRegressiveNN(self.latent_dim, [self.hidden_dim])

        iaf = dist.InverseAutoregressiveFlow(self.arn)
        pyro.module("{}_iaf".format(self.prefix), iaf)
        iaf_dist = dist.TransformedDistribution(dist.Normal(0., 1.).expand([self.latent_dim]), [iaf])
        return iaf_dist


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

    By default the mean vector is initialized to zero. To change this default behavior
    the user should call :func:`pyro.param` before beginning inference, e.g.::

        latent_dim = 10
        pyro.param("auto_loc", torch.randn(latent_dim))
    """

    def get_posterior(self, *args, **kwargs):
        """
        Returns a Delta posterior distribution for MAP inference.
        """
        loc = pyro.param("{}_loc".format(self.prefix), self._init_loc)
        return dist.Delta(loc).to_event(1)

    def laplace_approximation(self, *args, **kwargs):
        """
        Returns a :class:`AutoMultivariateNormal` instance whose posterior's `loc` and
        `scale_tril` are given by Laplace approximation.
        """
        guide_trace = poutine.trace(self).get_trace(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(self.model, trace=guide_trace)).get_trace(*args, **kwargs)
        loss = guide_trace.log_prob_sum() - model_trace.log_prob_sum()

        loc = pyro.param("{}_loc".format(self.prefix))
        H = hessian(loss, loc.unconstrained())
        cov = H.inverse()
        scale_tril = cov.cholesky()

        # calculate scale_tril from self.guide()
        scale_tril_name = "{}_scale_tril".format(self.prefix)
        pyro.param(scale_tril_name, scale_tril,
                   constraint=constraints.lower_cholesky)
        # force an update to scale_tril even if it already exists
        pyro.get_param_store()[scale_tril_name] = scale_tril

        gaussian_guide = AutoMultivariateNormal(self.model, prefix=self.prefix)
        gaussian_guide._setup_prototype(*args, **kwargs)
        return gaussian_guide


class AutoDiscreteParallel(AutoGuide):
    """
    A discrete mean-field guide that learns a latent discrete distribution for
    each discrete site in the model.
    """

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        model = config_enumerate(self.model)
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._discrete_sites = []
        self._cond_indep_stacks = {}
        self._plates = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            if site["infer"].get("enumerate") != "parallel":
                raise NotImplementedError('Expected sample site "{}" to be discrete and '
                                          'configured for parallel enumeration'.format(name))

            # collect discrete sample sites
            fn = site["fn"]
            Dist = type(fn)
            if Dist in (dist.Bernoulli, dist.Categorical, dist.OneHotCategorical):
                params = [("probs", fn.probs.detach().clone(), fn.arg_constraints["probs"])]
            else:
                raise NotImplementedError("{} is not supported".format(Dist.__name__))
            self._discrete_sites.append((site, Dist, params))

            # collect independence contexts
            self._cond_indep_stacks[name] = site["cond_indep_stack"]
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._plates[frame.name] = frame
                else:
                    raise NotImplementedError("AutoDiscreteParallel does not support sequential pyro.plate")

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()

        # enumerate discrete latent samples
        result = {}
        for site, Dist, param_spec in self._discrete_sites:
            name = site["name"]
            dist_params = {
                param_name: pyro.param("{}_{}_{}".format(self.prefix, name, param_name), param_init,
                                       constraint=param_constraint)
                for param_name, param_init, param_constraint in param_spec
            }
            discrete_dist = Dist(**dist_params)

            with ExitStack() as stack:
                for frame in self._cond_indep_stacks[name]:
                    stack.enter_context(plates[frame.name])
                result[name] = pyro.sample(name, discrete_dist, infer={"enumerate": "parallel"})

        return result


def mean_field_entropy(model, args, whitelist=None):
    """Computes the entropy of a model, assuming
    that the model is fully mean-field (i.e. all sample sites
    in the model are independent).

    The entropy is simply the sum of the entropies at the
    individual sites. If `whitelist` is not `None`, only sites
    listed in `whitelist` will have their entropies included
    in the sum. If `whitelist` is `None`, all non-subsample
    sites are included.
    """
    trace = poutine.trace(model).get_trace(*args)
    entropy = 0.
    for name, site in trace.nodes.items():
        if site["type"] == "sample":
            if not poutine.util.site_is_subsample(site):
                if whitelist is None or name in whitelist:
                    entropy += site["fn"].entropy()
    return entropy
