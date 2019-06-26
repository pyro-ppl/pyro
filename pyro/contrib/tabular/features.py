from __future__ import absolute_import, division, print_function

import math
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
        assert isinstance(name, str)
        self.name = name
        # We store a prototype rather than the device so that
        # when serializing Features via torch.save() and torch.load(),
        # the map_location kwarg is respected.
        self._prototype = torch.empty(1)

    @property
    def device(self):
        return self._prototype.device

    def __str__(self):
        return '{}("{}")'.format(type(self).__name__, self.name)

    def new_tensor(self, *args, **kwargs):
        kwargs.setdefault("device", self.device)
        return torch.tensor(*args, **kwargs)

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
        self._prototype = data.new_empty(1)


class Boolean(Feature):
    dtype = torch.float

    def sample_shared(self):
        loc = pyro.sample("{}_shared_loc".format(self.name),
                          dist.Normal(self.new_tensor(0.), self.new_tensor(2.)))
        scale = pyro.sample("{}_shared_scale".format(self.name),
                            dist.LogNormal(self.new_tensor(0.), self.new_tensor(1.)))
        return loc, scale

    def sample_group(self, shared):
        loc, scale = shared
        logits = pyro.sample("{}_group_logits".format(self.name), dist.Normal(loc, scale))
        if logits.dim() > 1:
            logits = logits.unsqueeze(-2)
        return logits

    def value_dist(self, group, component):
        logits = group
        logits = Vindex(logits)[..., component]
        return dist.Bernoulli(logits=logits)

    @torch.no_grad()
    def init(self, data):
        super(Boolean, self).init(data)

        assert data.dim() == 1
        mean = data.mean() * 0.98 + 0.01
        loc = mean.log() - (-mean).log1p()
        scale = data.new_tensor(1.)

        pyro.param("auto_{}_shared_loc".format(self.name), loc)
        pyro.param("auto_{}_shared_scale".format(self.name), scale,
                   constraint=constraints.positive)


class Discrete(Feature):
    dtype = torch.long

    def __init__(self, name, cardinality):
        super(Discrete, self).__init__(name)
        self.cardinality = cardinality

    def __str__(self):
        return '{}("{}", {})'.format(type(self).__name__, self.name, self.cardinality)

    def sample_shared(self):
        loc = pyro.sample("{}_shared_loc".format(self.name),
                          dist.Normal(self.new_tensor(0.), self.new_tensor(2.))
                              .expand([self.cardinality]).to_event(1))
        scale = pyro.sample("{}_shared_scale".format(self.name),
                            dist.LogNormal(self.new_tensor(0.), self.new_tensor(1.))
                                .expand([self.cardinality]).to_event(1))
        return loc, scale

    def sample_group(self, shared):
        loc, scale = shared
        logits = pyro.sample("{}_group_logits".format(self.name),
                             dist.Normal(loc, scale).to_event(1))
        if logits.dim() > 2:
            logits = logits.unsqueeze(-3)
        return logits

    def value_dist(self, group, component):
        logits = group
        logits = Vindex(logits)[..., component, :]
        return dist.Categorical(logits=logits)

    @torch.no_grad()
    def init(self, data):
        super(Discrete, self).init(data)

        assert data.dim() == 1
        counts = torch.zeros(self.cardinality, device=self.device)
        counts = counts.scatter_add(0, data.long(), torch.ones(data.shape, device=self.device))
        loc = (counts + 0.5).log()
        loc = loc - loc.logsumexp(-1, True)
        scale = loc.new_full(loc.shape, 2.)

        pyro.param("auto_{}_shared_loc".format(self.name), loc)
        pyro.param("auto_{}_shared_scale".format(self.name), scale,
                   constraint=constraints.positive)


class Real(Feature):
    dtype = torch.float

    def sample_shared(self):
        shape = pyro.sample("{}_shared_shape".format(self.name),
                            dist.LogNormal(self.new_tensor(0.), self.new_tensor(1.)))
        rate = pyro.sample("{}_shared_rate".format(self.name),
                           dist.LogNormal(self.new_tensor(0.), self.new_tensor(30.)))
        loc = pyro.sample("{}_shared_loc".format(self.name),
                          dist.Normal(self.new_tensor(0.), self.new_tensor(3.)))
        scale = pyro.sample("{}_shared_scale".format(self.name),
                            dist.LogNormal(self.new_tensor(0.), self.new_tensor(3.)))
        return shape, rate, loc, scale

    def sample_group(self, shared):
        shared_shape, shared_rate, shared_loc, shared_scale = shared
        precision = pyro.sample("{}_group_precision".format(self.name),
                                dist.Gamma(shared_shape, shared_rate))
        factor = shared_rate.sqrt()
        loc = pyro.sample("{}_group_loc".format(self.name),
                          dist.Normal(shared_loc * factor, shared_scale * factor))
        scale = precision.pow(-0.5)
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
        super(Real, self).init(data)

        assert data.dim() == 1
        data_scale = data.std(unbiased=False) + 1e-6
        rate = data_scale.pow(2) / math.exp(2.)
        shape = data.new_tensor(0.5)
        loc = data.mean() / data_scale * math.exp(1.)
        scale = data.new_tensor(2.).exp()

        # TODO add event_dim args for funsor backend
        pyro.param("auto_{}_shared_shape".format(self.name), shape,
                   constraint=constraints.positive)
        pyro.param("auto_{}_shared_rate".format(self.name), rate,
                   constraint=constraints.positive)
        pyro.param("auto_{}_shared_loc".format(self.name), loc)
        pyro.param("auto_{}_shared_scale".format(self.name), scale,
                   constraint=constraints.positive)

    class Summary(object):
        def __init__(self, feature, group):
            loc, scale = group
            self.count = loc.new_zeros(loc.shape)
            self.mean = loc.new_zeros(loc.shape)
            self.count_times_variance = loc.new_zeros(loc.shape)

        def scatter_update(self, component, data):
            count = torch.zeros_like(self.count).scatter_add_(component, 1.)
            mean = torch.zeros_like(self.mean).scatter_add_(component, data) / count
            self.count_times_var += self.count * (mean - self.mean).pow(2)
            self.mean += count / (count + self.count) * (mean - self.mean)
            self.count += count
            self.count_times_var.scatter_add_(component, (data - self.mean[component]).pow(2))

        def __imul__(self, scale):
            self.count *= scale
            self.count_times_variance *= scale

        def as_scaled_data(self):
            mean = self.count_times_mean / self.count
            scale = (self.count_times_variance / self.count).sqrt()
            data = torch.stack([mean - scale, mean + scale])
            scale = 0.5 * self.count
            return scale, data
