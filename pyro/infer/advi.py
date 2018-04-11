from __future__ import absolute_import, division, print_function

import weakref

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import sum_rightmost
from pyro.infer.enum import config_enumerate
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_traces_match

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


def _product(shape):
    """
    Computes the product of the dimensions of a given shape tensor
    """
    result = 1
    for size in shape:
        result *= size
    return result


class ADVIMaster(object):
    """
    Master ADVI class. This is a container for other ADVI strategies.
    """
    def __init__(self, model):
        self.parts = []
        self.base_model = model
        self.prototype_trace = None
        self._iaranges = {}
        self.iaranges = {}

    def _check_prototype(self, part_trace):
        for name, part_site in part_trace.nodes.items():
            if part_site["type"] != "sample":
                continue
            self_site = self.prototype_trace.nodes[name]
            assert part_site["fn"].batch_shape == self_site["fn"].batch_shape
            assert part_site["fn"].event_shape == self_site["fn"].event_shape
            assert part_site["value"].shape == self_site["value"].shape

    def add(self, part):
        assert isinstance(part, ADVISlave), type(part)
        self.parts.append(part)
        assert part.master is None
        part.master = weakref.ref(self)

    def model(self, *args, **kwargs):
        for part in self.parts:
            part.model(*args, **kwargs)
        return self.base_model(*args, **kwargs)

    def guide(self, *args, **kwargs):
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        # create all iaranges
        self.iaranges = {frame.name: pyro.iarange(frame.name, frame.size, dim=frame.dim)
                         for frame in sorted(self._iaranges.values())}

        # run slave guides
        for part in self.parts:
            part.guide(*args, **kwargs)

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        # self.prototype_trace = poutine.block(poutine.trace(self.base_model).get_trace)(*args, **kwargs)
        self.prototype_trace = poutine.block(poutine.trace(self.base_model).get_trace)(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)

        self._iaranges = {}
        for name, site in self.prototype_trace.nodes.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._iaranges[frame.name] = frame
                else:
                    raise NotImplementedError("ADVI does not support pyro.irange")


class ADVISlave(object):
    def __init__(self, model):
        self.master = None
        self.base_model = model
        self.prototype_trace = None

    def model(self, *args, **kwargs):
        """
        A wrapped model with the same ``*args, **kwargs`` as the base ``model``.
        """
        # wrap sample statement with a 0.0 poutine.scale to zero out unwanted score
        with poutine.scale(None, 0.0):
            self.sample_latent(*args, **kwargs)

        if self.master is None:
            # actual model sample statements shouldn't be zeroed out
            base_trace = poutine.trace(self.base_model).get_trace(*args, **kwargs)
            base_trace = prune_subsample_sites(base_trace)
            check_traces_match(base_trace, self.prototype_trace)
            return base_trace.nodes["_RETURN"]["value"]

    def sample_latent(*args, **kwargs):
        pass


class ADVIContinuous(ADVISlave):
    """
    Base class for implementations of Automatic Differentiation Variational Inference [1].

    Each derived class implements its own :meth:`sample_latent` method.

    Assumes model structure and latent dimension are fixed, and all latent
    variables are continuous.

    :param callable model: a Pyro model

    Reference:

    [1] 'Automatic Differentiation Variational Inference',
    Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M. Blei
    """
    def __init__(self, model):
        super(ADVIContinuous, self).__init__(model)

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        self.prototype_trace = poutine.block(poutine.trace(self.base_model).get_trace)(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._unconstrained_shapes = {}
        self._cond_indep_stacks = {}
        self._iaranges = {}
        for name, site in self.prototype_trace.nodes.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            # collect the shapes of unconstrained values, which may differ from the shapes of constrained values
            self._unconstrained_shapes[name] = biject_to(site["fn"].support).inv(site["value"]).shape

            # collect independence contexts
            self._cond_indep_stacks[name] = site["cond_indep_stack"]
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._iaranges[frame.name] = frame
                else:
                    raise NotImplementedError("ADVIContinuous does not support pyro.irange")

        self.latent_dim = sum(_product(shape) for shape in self._unconstrained_shapes.values())

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        raise NotImplementedError

    def _unpack_latent(self, latent):
        """
        Unpacks a packed latent tensor, iterating over tuples of the form::

            (site, unconstrained_value)
        """
        pos = 0
        for name, site in self.prototype_trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                unconstrained_shape = self._unconstrained_shapes[name]
                size = _product(unconstrained_shape)
                unconstrained_value = latent[pos:pos + size].view(unconstrained_shape)
                yield site, unconstrained_value
                pos += size
        assert pos == len(latent)

    def guide(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dictionary mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        latent = self.sample_latent(*args, **kwargs)

        # create all iaranges
        if self.master is None:
            iaranges = {frame.name: pyro.iarange(frame.name, frame.size, dim=frame.dim)
                        for frame in sorted(self._iaranges.values())}
        else:
            iaranges = self.master().iaranges

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
                    stack.enter_context(iaranges[frame.name])
                result[name] = pyro.sample(name, delta_dist)

        return result


class ADVIMultivariateNormal(ADVIContinuous):
    """
    This implementation of ADVI uses a Cholesky factorization of a Multivariate
    Normal distribution to construct a guide over the entire latent space. The
    guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        advi = ADVIMultivariateNormal(model)
        svi = SVI(advi.model, advi.guide, ...)

    By default the mean vector is initialized to zero and the Cholesky factor
    is initialized to the identity.  To change this default behavior the user
    should call :func:`pyro.param` before beginning inference, e.g.::

        latent_dim = 10
        pyro.param("advi_loc", torch.randn(latent_dim))
        pyro.param("advi_scale_tril", torch.tril(torch.rand(latent_dim)),
                   constraint=constraints.lower_cholesky)
    """
    def sample_latent(self, *args, **kwargs):
        """
        Samples the (single) multivariate normal latent used in the advi guide.
        """
        loc = pyro.param("advi_loc", torch.zeros(self.latent_dim))
        scale_tril = pyro.param("advi_scale_tril", torch.eye(self.latent_dim),
                                constraint=constraints.lower_cholesky)
        return pyro.sample("_advi_latent", dist.MultivariateNormal(loc, scale_tril=scale_tril))

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dictionary mapping sample site name to median tensor.
        :rtype: dict
        """
        latent = pyro.param("advi_loc")
        return {site["name"]: biject_to(site["fn"].support)(unconstrained_value)
                for site, unconstrained_value in self._unpack_latent(latent)}

    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(advi.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dictionary mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        loc = pyro.param("advi_loc")
        scale = pyro.param("advi_scale_tril").diag()
        quantiles = loc.new_tensor(quantiles).unsqueeze(-1)
        latents = dist.Normal(loc, scale).icdf(quantiles)

        result = {}
        for latent in latents:
            for site, unconstrained_value in self._unpack_latent(latent):
                result.setdefault(site["name"], []).append(biject_to(site["fn"].support)(unconstrained_value))
        return result


class ADVIDiagonalNormal(ADVIContinuous):
    """
    This implementation of ADVI uses a Normal distribution with a diagonal
    covariance matrix to construct a guide over the entire latent space. The
    guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        advi = ADVIDiagonalNormal(model)
        svi = SVI(advi.model, advi.guide, ...)

    By default the mean vector is initialized to zero and the scale is
    initialized to the identity.  To change this default behavior the user
    should call :func:`pyro.param` before beginning inference, e.g.::

        latent_dim = 10
        pyro.param("advi_loc", torch.randn(latent_dim))
        pyro.param("advi_scale", torch.ones(latent_dim),
                   constraint=constraints.positive)
    """
    def sample_latent(self, *args, **kwargs):
        """
        Samples the (single) diagnoal normal latent used in the advi guide.
        """
        loc = pyro.param("advi_loc", torch.zeros(self.latent_dim))
        scale = pyro.param("advi_scale", torch.ones(self.latent_dim),
                           constraint=constraints.positive)
        return pyro.sample("_advi_latent", dist.Normal(loc, scale).reshape(extra_event_dims=1))

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dictionary mapping sample site name to median tensor.
        :rtype: dict
        """
        latent = pyro.param("advi_loc")
        return {site["name"]: biject_to(site["fn"].support)(unconstrained_value)
                for site, unconstrained_value in self._unpack_latent(latent)}

    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(advi.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dictionary mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        loc = pyro.param("advi_loc")
        scale = pyro.param("advi_scale")
        quantiles = loc.new_tensor(quantiles).unsqueeze(-1)
        latents = dist.Normal(loc, scale).icdf(quantiles)

        result = {}
        for latent in latents:
            for site, unconstrained_value in self._unpack_latent(latent):
                result.setdefault(site["name"], []).append(biject_to(site["fn"].support)(unconstrained_value))
        return result


class ADVIDiscreteParallel(ADVISlave):
    def __init__(self, model):
        super(ADVIDiscreteParallel, self).__init__(model)

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        model = config_enumerate(self.base_model, default="parallel")
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)
        if self.master is not None:
            self.master()._check_prototype(self.prototype_trace)

        self._discrete_sites = []
        self._cond_indep_stacks = {}
        self._iaranges = {}
        for name, site in self.prototype_trace.nodes.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
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
                    self._iaranges[frame.name] = frame
                else:
                    raise NotImplementedError("ADVI does not support pyro.irange")

    def guide(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dictionary mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        # create all iaranges
        if self.master is None:
            iaranges = {frame.name: pyro.iarange(frame.name, frame.size, dim=frame.dim)
                        for frame in sorted(self._iaranges.values())}
        else:
            iaranges = self.master().iaranges

        # enumerate discrete latent samples
        result = {}
        for site, Dist, param_spec in self._discrete_sites:
            name = site["name"]
            dist_params = {
                param_name: pyro.param("advi_discrete_{}_{}".format(name, param_name), param_init,
                                       constraint=param_constraint)
                for param_name, param_init, param_constraint in param_spec
            }
            discrete_dist = Dist(**dist_params)

            with ExitStack() as stack:
                for frame in self._cond_indep_stacks[name]:
                    stack.enter_context(iaranges[frame.name])
                result[name] = pyro.sample(name, discrete_dist, infer={"enumerate": "parallel"})

        return result
