from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import torch
from six import add_metaclass
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.ops.indexing import Vindex


@add_metaclass(ABCMeta)
class Feature(object):
    """
    A hierarchical mixture model for a single observed feature type.

    Feature models are intended as an adapter between latent categorical
    variables and observed variables of various other datatypes.  Feature
    models are agnostic to component weight models, which may e.g. be driven
    by a neural network. Thus users must sample component membership from a
    user-provided distribution.

    Example usage::

        # User must provide a membership distribution.
        num_components = 10
        weights = pyro.param("component_weights",
                             torch.ones(num_components) / num_components,
                             constraint=constraints.simplex)
        membership_dist = dist.Categorical(weights)

        # Feature objects adapt the component id to a given feature type.
        f = MyFeature("foo")
        f.init(data)  # (optional) initializes AutoGuide parameters
        shared = f.sample_shared()
        with pyro.plate("components", num_components):
            group = f.sample_group(shared)  # broadcasts to each component
        with pyro.plate("data", len(data)):
            component = pyro.sample("component", membership_dist)
            pyro.sample("obs", f.value_dist(group, component), obs=data)

    :param str name: The name of this feature, used as a prefix for
        :func:`pyro.sample` statements inside :meth:`sample_shared` and
        :meth:`sample_group`.
    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '{}("{}")'.format(type(self).__name__, self.name)

    @abstractmethod
    def sample_shared(self):
        """
        Samples parameters of this feature model that are shared by all mixture
        components.

        :returns: an opaque result to be passed to :meth:`sample_group`.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_group(self, shared):
        """
        Samples per-component parameters of this feature model. This should be
        called inside a vectorized :class:`~pyro.plate` over the desired
        number of components. This is intended to be executed inside an
        :class:`~pyro.plate` over the number of components.

        :returns: an opaque result to be passed to :meth:`value_dist`.
        """
        raise NotImplementedError

    @abstractmethod
    def value_dist(self, group, component):
        """
        Constructs a :class:`~pyro.distributions.TorchDistribution`
        object conditioned on the given component index.

        :param group: an opaque result of :meth:`sample_group`
        :param torch.LongTensor component: a component id or batch of component
            ids
        :returns: a component distribution object
        :rtype: pyro.distributions.TorchDistribution
        """
        raise NotImplementedError

    @torch.no_grad()
    def init(self, data):
        """
        Heuristically initializes :class:`~pyro.contrib.autoguide.AutoDelta`
        parameters for shared variables based on example data.

        This returns nothing; results are saved in the Pyro param store.

        This method is optional. For subclasses that do not implement this
        method, parameters will be sampled from the prior.

        :param torch.Tensor data: A dataset or subsample of data.
        """
        assert data.dim() == 1


class Boolean(Feature):
    def sample_shared(self):
        loc = pyro.sample("{}_loc".format(self.name), dist.Normal(0., 2.))
        scale = pyro.sample("{}_scale".format(self.name), dist.LogNormal(0., 1.))
        return loc, scale

    def sample_group(self, shared):
        loc, scale = shared
        logits = pyro.sample("{}_logits".format(self.name), dist.Normal(loc, scale))
        if logits.dim() > 1:
            logits = logits.unsqueeze(-2)
        return logits

    def value_dist(self, group, component):
        logits = group
        logits = Vindex(logits)[..., component]
        return dist.Bernoulli(logits=logits)

    @torch.no_grad()
    def init(self, data):
        assert data.dim() == 1
        mean = data.mean() * 0.98 + 0.01
        loc = mean.log() - (-mean).log1p()
        scale = data.new_tensor(1.)

        pyro.param("auto_{}_loc".format(self.name), loc)
        pyro.param("auto_{}_scale".format(self.name), scale,
                   constraint=constraints.positive)


class Discrete(Feature):
    def __init__(self, name, cardinality):
        super(Discrete, self).__init__(name)
        self.cardinality = cardinality

    def sample_shared(self):
        loc = pyro.sample("{}_loc".format(self.name),
                          dist.Normal(0., 2.).expand([self.cardinality]).to_event(1))
        scale = pyro.sample("{}_scale".format(self.name),
                            dist.LogNormal(0., 1.).expand([self.cardinality]).to_event(1))
        return loc, scale

    def sample_group(self, shared):
        loc, scale = shared
        logits = pyro.sample("{}_logits".format(self.name),
                             dist.Normal(loc, scale).to_event(1))
        if logits.dim() > 1:
            logits = logits.unsqueeze(-3)
        return logits

    def value_dist(self, group, component):
        logits = group
        logits = Vindex(logits)[..., component, :]
        return dist.Categorical(logits=logits)

    @torch.no_grad()
    def init(self, data):
        assert data.dim() == 1
        counts = torch.zeros(self.cardinality, device=data.device)
        counts = counts.scatter_add(0, data, torch.ones(data.shape, device=data.device))
        loc = (counts + 0.5) / (len(data) + 0.5 * len(counts))
        scale = loc.new_ones(loc.shape)

        pyro.param("auto_{}_loc".format(self.name), loc)
        pyro.param("auto_{}_scale".format(self.name), scale,
                   constraint=constraints.positive)


class Real(Feature):
    def sample_shared(self):
        scale_loc = pyro.sample("{}_scale_loc".format(self.name), dist.Normal(0., 10.))
        scale_scale = pyro.sample("{}_scale_scale".format(self.name), dist.LogNormal(0., 1.))
        loc_loc = pyro.sample("{}_loc_loc".format(self.name), dist.Normal(0., 3.))
        loc_scale = pyro.sample("{}_loc_scale".format(self.name), dist.LogNormal(0., 3.))
        return scale_loc, scale_scale, loc_loc, loc_scale

    def sample_group(self, shared):
        scale_loc, scale_scale, loc_loc, loc_scale = shared
        scale = pyro.sample("{}_scale".format(self.name),
                            dist.LogNormal(scale_loc, scale_scale))
        loc = pyro.sample("{}_loc".format(self.name),
                          dist.Normal(loc_loc * scale, loc_scale * scale))
        if loc.dim() > 1:
            loc = loc.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        return loc, scale

    def value_dist(self, group, component):
        loc, scale = group
        loc = Vindex(loc)[..., component]
        scale = Vindex(scale)[..., component]
        return dist.Normal(loc, scale)

    @torch.no_grad()
    def init(self, data):
        assert data.dim() == 1
        data_std = data.std(unbiased=False) + 1e-6
        scale_loc = data_std.log()
        scale_scale = data.new_tensor(1.)
        loc_loc = data.mean() / data_std
        loc_scale = data.new_tensor(1.)

        pyro.param("auto_{}_scale_loc".format(self.name), scale_loc)
        pyro.param("auto_{}_scale_scale".format(self.name), scale_scale,
                   constraint=constraints.positive)
        pyro.param("auto_{}_loc_loc".format(self.name), loc_loc)
        pyro.param("auto_{}_loc_scale".format(self.name), loc_scale,
                   constraint=constraints.positive)
