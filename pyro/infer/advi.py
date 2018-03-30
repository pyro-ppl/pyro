from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import sum_rightmost
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


class ADVI(object):
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
        self.prototype_trace = None
        self.base_model = model

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        self.prototype_trace = poutine.block(poutine.trace(self.base_model).get_trace)(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)

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
                    raise NotImplementedError("ADVI does not support pyro.irange")

        self.latent_dim = sum(_product(shape) for shape in self._unconstrained_shapes.values())

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        raise NotImplementedError

    def guide(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dictionary mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        result = {}
        latent = self.sample_latent(*args, **kwargs)

        # create all iaranges
        iaranges = {frame.name: pyro.iarange(frame.name, frame.size, dim=frame.dim)
                    for frame in sorted(self._iaranges.values())}

        # unpack latent samples
        pos = 0
        for name, site in self.prototype_trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                unconstrained_shape = self._unconstrained_shapes[name]
                size = _product(unconstrained_shape)
                unconstrained_value = latent[pos:pos + size].view(unconstrained_shape)
                pos += size
                transform = biject_to(site["fn"].support)
                value = transform(unconstrained_value)
                log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
                log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + site["fn"].event_dim)
                delta_dist = dist.Delta(value, log_density=log_density, event_dim=site["fn"].event_dim)

                with ExitStack() as stack:
                    for frame in self._cond_indep_stacks[name]:
                        stack.enter_context(iaranges[frame.name])
                    result[name] = pyro.sample(name, delta_dist)

        assert pos == len(latent)

        return result

    def model(self, *args, **kwargs):
        """
        A wrapped model with the same ``*args, **kwargs`` as the base ``model``.
        """
        # wrap sample statement with a 0.0 poutine.scale to zero out unwanted score
        with poutine.scale("advi_scope", 0.0):
            self.sample_latent(*args, **kwargs)
        # actual model sample statements shouldn't be zeroed out
        base_trace = poutine.trace(self.base_model).get_trace(*args, **kwargs)
        base_trace = prune_subsample_sites(base_trace)
        check_traces_match(base_trace, self.prototype_trace)
        return base_trace.nodes["_RETURN"]["value"]


class ADVIMultivariateNormal(ADVI):
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


class ADVIDiagonalNormal(ADVI):
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
