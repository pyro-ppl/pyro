# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch

import torch

import pyro.distributions as dist
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample


class PrefixReplayMessenger(Messenger):
    """
    EXPERIMENTAL Given a trace of training data, replay a model with batched
    sites extended to include both training and forecast time, using the guide
    trace for the training prefix and samples from the prior on the forecast
    postfix.

    :param trace: a guide trace.
    :type trace: ~pyro.poutine.trace_struct.Trace
    """
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def _pyro_post_sample(self, msg):
        if site_is_subsample(msg):
            return

        name = msg["name"]
        if name not in self.trace:
            return

        model_value = msg["value"]
        guide_value = self.trace.nodes[name]["value"]
        if model_value.shape == guide_value.shape:
            msg["value"] = guide_value
            return

        assert model_value.dim() >= 2
        assert model_value.dim() == guide_value.dim()
        assert model_value.shape[:-2] == guide_value.shape[:-2]
        assert model_value.size(-2) > guide_value.size(-2)
        assert model_value.size(-1) == guide_value.size(-1)
        assert any(f.name == "time" for f in msg["cond_indep_stack"])
        split = guide_value.size(-2)
        msg["value"] = torch.cat([guide_value, model_value[..., split:, :]], dim=-2)


# ----------------------------------------------------------------------------
# The pattern-matching code in the remainder of ths file could be eventually
# replace by much simpler Funsor logic.

UNIVARIATE_DISTS = {
    dist.Cauchy: ("loc", "scale"),
    dist.Laplace: ("loc", "scale"),
    dist.Normal: ("loc", "scale"),
    dist.Stable: ("stability", "skew", "scale", "loc"),
    dist.StudentT: ("df", "loc", "scale"),
}


@singledispatch
def prefix_condition(d, data):
    """
    EXPERIMENTAL Given a distribution ``d`` of shape ``batch_shape + (t+f, d)``
    and data ``x`` of shape ``batch_shape + (t, d)``, find a conditional
    distribution of shape ``batch_shape + (f, d)``. Typically ``t`` is the
    number of training time steps, ``f`` is the number of forecast time steps,
    and ``d`` is the data dimension.

    :param d: a distribution with ``len(d.shape()) >= 2``
    :type d: ~pyro.distributions.Distribution
    :param data: data of dimension at least 2.
    :type data: ~torch.Tensor
    """
    try:
        return d.prefix_condition(data)
    except AttributeError:
        raise NotImplementedError("prefix_condition() does not suport {}".format(type(d)))


@prefix_condition.register(dist.Independent)
def _(d, data):
    base_dist = prefix_condition(d.base_dist, data)
    return base_dist.to_event(d.reinterpreted_batch_ndims)


def _prefix_condition_univariate(d, data):
    t = data.size(-2)
    params = [getattr(d, name)[..., t:, :]
              for name in UNIVARIATE_DISTS[type(d)]]
    return type(d)(*params)


for _type in UNIVARIATE_DISTS:
    prefix_condition.register(_type)(_prefix_condition_univariate)


@singledispatch
def reshape_batch(d, batch_shape):
    """
    EXPERIMENTAL Given a distribution ``d``, reshape to different batch shape
    of same number of elements.

    This is typically used to move the the rightmost batch dimension "time" to
    an event dimension, while preserving the positions of other batch
    dimensions.

    :param d: A distribution.
    :type d: ~pyro.distributions.Distribution
    :param tuple batch_shape: A new batch shape.
    :returns: A distribution with the same type but given batch shape.
    :rtype: ~pyro.distributions.Distribution
    """
    raise NotImplementedError


@reshape_batch.register(dist.Independent)
def _(d, batch_shape):
    base_shape = batch_shape + d.event_shape[:d.reinterpreted_batch_ndims]
    base_dist = reshape_batch(d.base_dist, base_shape)
    return base_dist.to_event(d.reinterpreted_batch_ndims)


def _reshape_batch_univariate(d, batch_shape):
    params = [getattr(d, name).reshape(batch_shape)
              for name in UNIVARIATE_DISTS[type(d)]]
    return type(d)(*params)


for _type in UNIVARIATE_DISTS:
    reshape_batch.register(_type)(_reshape_batch_univariate)


@reshape_batch.register(dist.GaussianHMM)
def _(d, batch_shape):
    # TODO this could save computation by less eagerly expanding
    new = d._get_checked_instance(dist.GaussianHMM)
    new.hidden_dim = d.hidden_dim
    new.obs_dim = d.obs_dim
    new._init = d._init.expand(d.batch_shape).reshape(batch_shape)
    new._obs = d._obs.expand(d.batch_shape + (-1,)).reshape(batch_shape + d._obs.batch_shape[-1:])
    new._trans = d._trans.expand(d.batch_shape + (-1,)).reshape(batch_shape + d._trans.batch_shape[-1:])
    super(dist.GaussianHMM, new).__init__(d.duration, batch_shape, d.event_shape,
                                          validate_args=d._validate_args)
    return new


@reshape_batch.register(dist.LinearHMM)
def _(d, batch_shape):
    # TODO this could save computation by less eagerly expanding
    return dist.LinearHMM(
        reshape_batch(d.initial_dist.expand(d.batch_shape), batch_shape),
        d.transition_matrix
         .expand(d.batch_shape + (-1, d.hidden_dim, d.hidden_dim))
         .reshape(batch_shape + (-1, d.hidden_dim, d.hidden_dim)),
        reshape_batch(d.transition_dist.expand(d.batch_shape + (-1,)), batch_shape + (-1,)),
        d.observation_matrix
         .expand(d.batch_shape + (-1, d.hidden_dim, d.obs_dim))
         .reshape(batch_shape + (-1, d.hidden_dim, d.obs_dim)),
        reshape_batch(d.transition_dist.expand(d.batch_shape + (-1,)), batch_shape + (-1,)),
        duration=d.duration)
