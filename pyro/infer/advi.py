from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_traces_match


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
        self.latent_dim = sum(site["value"].view(-1).size(0)
                              for site in self.prototype_trace.nodes.values()
                              if site["type"] == "sample" and not site["is_observed"])

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        raise NotImplementedError

    def guide(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        latent = self.sample_latent(*args, **kwargs)
        pos = 0
        for name, site in self.prototype_trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                shape = site["fn"].shape()
                size = _product(shape)
                unconstrained_value = latent[pos:pos + size].view(shape)
                pos += size
                delta_dist = dist.TransformedDistribution(dist.Delta(unconstrained_value),
                                                          transform_to(site["fn"].support))
                pyro.sample(name, delta_dist.reshape(extra_event_dims=site["fn"].event_dim))
        assert pos == len(latent)

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
        pyro.param("advi_cholesky_factor", torch.tril(torch.rand(latent_dim)),
                   constraint=constraints.lower_cholesky)
    """
    def sample_latent(self, *args, **kwargs):
        """
        Samples the (single) multivariate normal latent used in the advi guide.
        """
        loc = pyro.param("advi_loc", torch.zeros(self.latent_dim))
        lower_cholesky = pyro.param("advi_lower_cholesky", torch.eye(self.latent_dim),
                                    constraint=constraints.lower_cholesky)
        cov = torch.mm(lower_cholesky, lower_cholesky.t())
        # TODO use Multivariate normal that consumes L directly
        return pyro.sample("_advi_latent", dist.MultivariateNormal(loc, cov))


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
