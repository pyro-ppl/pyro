# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch

import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistributionMixin


def order(x, batch_dims):
    batch_shape = set(getattr(x, "dims", ()))
    event_shape = x.shape
    if batch_shape:
        x = x.order(*(dim for dim in batch_dims if dim in batch_shape))
        x = x.reshape(
            tuple(dim.size if dim in batch_shape else 1 for dim in batch_dims)
            + event_shape
        )
    return x


def index_select(input, dim, index):
    return input.order(dim)[index]


class NamedDistribution(TorchDistributionMixin):
    dist_class: dist.Distribution

    def __init__(self, *args, **kwargs) -> None:
        ast_fields = inspect.getfullargspec(self.dist_class.__init__)[0][1:]
        kwargs.update(zip(ast_fields, args))
        self.batch_dims = tuple(
            set.union(
                *[
                    set(getattr(kwargs[k], "dims", ()))
                    for k in kwargs
                    if k in self.dist_class.arg_constraints
                ]
            )
        )
        for k in self.dist_class.arg_constraints:
            if k in kwargs:
                kwargs[k] = order(kwargs[k], self.batch_dims)
        self.base_dist = self.dist_class(**kwargs)
        self.sample_shape = torch.Size()

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @property
    def batch_shape(self):
        return self.batch_dims

    @property
    def event_shape(self):
        return self.base_dist.event_shape

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(self.sample_shape + sample_shape)[self.batch_dims]

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(self.sample_shape + sample_shape)[self.batch_dims]

    def log_prob(self, value):
        value_dims = set(getattr(value, "dims", ()))
        extra_dims = tuple(value_dims - set(self.batch_dims))
        value = order(value, extra_dims + self.batch_dims)
        return self.base_dist.log_prob(value)[extra_dims + self.batch_dims]

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new :class:`ExpandedDistribution` instance with batch
        dimensions expanded to `batch_shape`.

        :param tuple batch_shape: batch shape to expand to.
        :param _instance: unused argument for compatibility with
            :meth:`torch.distributions.Distribution.expand`
        :return: an instance of `ExpandedDistribution`.
        :rtype: :class:`ExpandedDistribution`
        """
        for dim in batch_shape:
            if dim not in set(self.batch_dims):
                self.batch_dims = self.batch_dims + (dim,)
                self.sample_shape = self.sample_shape + (dim.size,)
        return self

    def enumerate_support(self, expand=False):
        samples = self.base_dist.enumerate_support(expand=False)
        return samples


# class NamedDistributionMeta(type):
#     pass
# def __call__(cls, *args, **kwargs):


def make_dist(backend_dist_class):

    dist_class = type(
        backend_dist_class.__name__,
        (NamedDistribution,),
        {"dist_class": backend_dist_class},
    )
    return dist_class


Normal = make_dist(dist.Normal)
Categorical = make_dist(dist.Categorical)
LogNormal = make_dist(dist.LogNormal)
Dirichlet = make_dist(dist.Dirichlet)
