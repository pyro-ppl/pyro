# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch

import torch
from torch.distributions import transform_to, transforms

import pyro.distributions as dist
from pyro.infer.reparam import HaarReparam, DiscreteCosineReparam
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample
from pyro.primitives import get_param_store


def time_reparam_dct(msg):
    """
    EXPERIMENTAL Configures ``poutine.reparam()`` to use a ``HaarReparam`` for
    all sites inside the ``time`` plate.
    """
    if msg["is_observed"]:
        return
    for frame in msg["cond_indep_stack"]:
        if frame.name == "time":
            dim = frame.dim - msg["fn"].event_dim
            return HaarReparam(dim=dim, experimental_allow_batch=True)


def time_reparam_haar(msg):
    """
    EXPERIMENTAL Configures ``poutine.reparam()`` to use a ``DiscreteCosineReparam`` for
    all sites inside the ``time`` plate.
    """
    if msg["is_observed"]:
        return
    for frame in msg["cond_indep_stack"]:
        if frame.name == "time":
            dim = frame.dim - msg["fn"].event_dim
            return DiscreteCosineReparam(dim=dim, experimental_allow_batch=True)


class MarkDCTParamMessenger(Messenger):
    """
    EXPERIMENTAL Messenger to mark DCT dimension of parameter, for use with
    :class:`pyro.optim.optim.DCTAdam`.

    :param str name: The name of the plate along which to apply discrete cosine
        transforms on gradients.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def _postprocess_message(self, msg):
        if msg["type"] != "param":
            return
        event_dim = msg["kwargs"].get("event_dim")
        if event_dim is None:
            return
        for frame in msg["cond_indep_stack"]:
            if frame.name == self.name:
                value = msg["value"]
                event_dim += value.unconstrained().dim() - value.dim()
                value.unconstrained()._pyro_dct_dim = frame.dim - event_dim
                return


class PrefixWarmStartMessenger(Messenger):
    """
    EXPERIMENTAL Assuming the global param store has been populated with params
    defined on a short time window, re-initialize by splicing old params with
    new initial params defined on a longer time window.
    """
    def _pyro_param(self, msg):
        store = get_param_store()
        name = msg["name"]
        if name not in store:
            return

        if len(msg["args"]) >= 2:
            new = msg["args"][1]
        elif "init_tensor" in msg["kwargs"]:
            new = msg["kwargs"]["init_tensor"]
        else:
            return  # no init tensor specified

        if callable(new):
            new = new()
        old = store[name]
        assert new.dim() == old.dim()
        if new.shape == old.shape:
            return

        # Splice old (warm start) and new (init) tensors.
        # This only works for time-homogeneous constraints.
        t = transform_to(store._constraints[name])
        new = t.inv(new)
        old = t.inv(old)
        for dim in range(new.dim()):
            if new.size(dim) != old.size(dim):
                break
        assert new.size(dim) > old.size(dim)
        assert new.shape[dim + 1:] == old.shape[dim + 1:]
        split = old.size(dim)
        index = (slice(None),) * dim + (slice(split, None),)
        new = torch.cat([old, new[index]], dim=dim)
        store[name] = t(new)


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

        # Search for a single dim with mismatched size.
        assert model_value.dim() == guide_value.dim()
        for dim in range(model_value.dim()):
            if model_value.size(dim) != guide_value.size(dim):
                break
        assert model_value.size(dim) > guide_value.size(dim)
        assert model_value.shape[dim + 1:] == guide_value.shape[dim + 1:]
        split = guide_value.size(dim)
        index = (slice(None),) * dim + (slice(split, None),)
        msg["value"] = torch.cat([guide_value, model_value[index]], dim=dim)


class PrefixConditionMessenger(Messenger):
    """
    EXPERIMENTAL Given a prefix of t-many observations, condition a (t+f)-long
    distribution on the observations, converting it to an f-long distribution.

    :param dict data: A dict mapping site name to tensors of observations.
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _pyro_sample(self, msg):
        if msg["name"] not in self.data:
            return

        assert msg["value"] is None
        data = self.data[msg["name"]]
        msg["fn"] = prefix_condition(msg["fn"], data)


# ----------------------------------------------------------------------------
# The pattern-matching code in the remainder of this file could be eventually
# replace by much simpler Funsor logic.

UNIVARIATE_DISTS = {
    dist.Bernoulli: ("logits",),
    dist.Beta: ("concentration1", "concentration0"),
    dist.BetaBinomial: ("concentration1", "concentration0"),
    dist.Cauchy: ("loc", "scale"),
    dist.Dirichlet: ("concentration",),
    dist.DirichletMultinomial: ("concentration",),
    dist.Exponential: ("rate",),
    dist.Gamma: ("concentration", "rate"),
    dist.GammaPoisson: ("concentration", "rate"),
    dist.Geometric: ("logits",),
    dist.InverseGamma: ("concentration", "rate"),
    dist.Laplace: ("loc", "scale"),
    dist.LogNormal: ("loc", "scale"),
    dist.NegativeBinomial: ("total_count", "logits"),
    dist.Normal: ("loc", "scale"),
    dist.Poisson: ("rate",),
    dist.Stable: ("stability", "skew", "scale", "loc"),
    dist.StudentT: ("df", "loc", "scale"),
    dist.ZeroInflatedPoisson: ("rate", "gate"),
    dist.ZeroInflatedNegativeBinomial: ("total_count", "logits", "gate"),
}


@singledispatch
def prefix_condition(d, data):
    """
    EXPERIMENTAL Given a distribution ``d`` of shape ``batch_shape + (t+f, d)``
    and data ``x`` of shape ``batch_shape + (t, d)``, compute a conditional
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
    except AttributeError as e:
        raise NotImplementedError("prefix_condition() does not suport {}".format(type(d))) from e


@prefix_condition.register(dist.Independent)
def _(d, data):
    base_dist = prefix_condition(d.base_dist, data)
    return base_dist.to_event(d.reinterpreted_batch_ndims)


@prefix_condition.register(dist.IndependentHMM)
def _(d, data):
    base_data = data.transpose(-1, -2).unsqueeze(-1)
    base_dist = prefix_condition(d.base_dist, base_data)
    return dist.IndependentHMM(base_dist)


@prefix_condition.register(dist.FoldedDistribution)
def _(d, data):
    base_dist = prefix_condition(d.base_dist, data)
    return dist.FoldedDistribution(base_dist)


def _prefix_condition_univariate(d, data):
    t = data.size(-2)
    params = {name: getattr(d, name)[..., t:, :]
              for name in UNIVARIATE_DISTS[type(d)]}
    return type(d)(**params)


for _type in UNIVARIATE_DISTS:
    prefix_condition.register(_type)(_prefix_condition_univariate)


@prefix_condition.register(dist.MultivariateNormal)
def _(d, data):
    t = data.size(-2)
    loc = d.loc[..., t:, :]
    scale_tril = d.scale_tril[..., t:, :, :]
    return dist.MultivariateNormal(loc, scale_tril=scale_tril)


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
    raise NotImplementedError("reshape_batch() does not suport {}".format(type(d)))


@reshape_batch.register(dist.Independent)
def _(d, batch_shape):
    base_shape = batch_shape + d.event_shape[:d.reinterpreted_batch_ndims]
    base_dist = reshape_batch(d.base_dist, base_shape)
    return base_dist.to_event(d.reinterpreted_batch_ndims)


@reshape_batch.register(dist.IndependentHMM)
def _(d, batch_shape):
    base_shape = batch_shape + d.event_shape[-1:]
    base_dist = reshape_batch(d.base_dist, base_shape)
    return dist.IndependentHMM(base_dist)


@reshape_batch.register(dist.FoldedDistribution)
def _(d, batch_shape):
    base_dist = reshape_batch(d.base_dist, batch_shape)
    return dist.FoldedDistribution(base_dist)


def _reshape_batch_univariate(d, batch_shape):
    batch_shape = batch_shape + (-1,) * d.event_dim
    params = {name: getattr(d, name).reshape(batch_shape)
              for name in UNIVARIATE_DISTS[type(d)]}
    return type(d)(**params)


for _type in UNIVARIATE_DISTS:
    reshape_batch.register(_type)(_reshape_batch_univariate)


@reshape_batch.register(dist.MultivariateNormal)
def _(d, batch_shape):
    dim = d.event_shape[0]
    loc = d.loc.reshape(batch_shape + (dim,))
    scale_tril = d.scale_tril.reshape(batch_shape + (dim, dim))
    return dist.MultivariateNormal(loc, scale_tril=scale_tril)


@reshape_batch.register(dist.GaussianHMM)
def _(d, batch_shape):
    init = d._init
    if init.batch_shape:
        init = init.expand(d.batch_shape)
        init = init.reshape(batch_shape)

    trans = d._trans
    if len(trans.batch_shape) > 1:
        trans = trans.expand(d.batch_shape + (-1,))
        trans = trans.reshape(batch_shape + (-1,))

    obs = d._obs
    if len(obs.batch_shape) > 1:
        obs = obs.expand(d.batch_shape + (-1,))
        obs = obs.reshape(batch_shape + (-1,))

    new = d._get_checked_instance(dist.GaussianHMM)
    new.hidden_dim = d.hidden_dim
    new.obs_dim = d.obs_dim
    new._init = init
    new._trans = trans
    new._obs = obs
    super(dist.GaussianHMM, new).__init__(d.duration, batch_shape, d.event_shape,
                                          validate_args=d._validate_args)
    return new


@reshape_batch.register(dist.LinearHMM)
def _(d, batch_shape):
    init_dist = reshape_batch(d.initial_dist, batch_shape)
    trans_mat = d.transition_matrix.reshape(batch_shape + (-1, d.hidden_dim, d.hidden_dim))
    trans_dist = reshape_batch(d.transition_dist, batch_shape + (-1,))
    obs_mat = d.observation_matrix.reshape(batch_shape + (-1, d.hidden_dim, d.obs_dim))
    obs_dist = reshape_batch(d.observation_dist, batch_shape + (-1,))

    new = d._get_checked_instance(dist.LinearHMM)
    new.hidden_dim = d.hidden_dim
    new.obs_dim = d.obs_dim
    new.initial_dist = init_dist
    new.transition_matrix = trans_mat
    new.transition_dist = trans_dist
    new.observation_matrix = obs_mat
    new.observation_dist = obs_dist
    transforms = []
    for transform in d.transforms:
        assert type(transform) in UNIVARIATE_TRANSFORMS, \
            "Currently, reshape_batch only supports AbsTransform, " + \
            "ExpTransform, SigmoidTransform transform"
        old_shape = d.observation_dist.shape()
        new_shape = obs_dist.shape()
        transforms.append(reshape_transform_batch(transform, old_shape, new_shape))
    new.transforms = transforms
    super(dist.LinearHMM, new).__init__(d.duration, batch_shape, d.event_shape,
                                        validate_args=d._validate_args)
    return new


UNIVARIATE_TRANSFORMS = {
    transforms.AbsTransform: (),
    transforms.AffineTransform: ("loc", "scale"),
    transforms.ExpTransform: (),
    transforms.PowerTransform: ("exponent",),
    transforms.SigmoidTransform: (),
}


@singledispatch
def reshape_transform_batch(t, old_shape, new_shape):
    """
    EXPERIMENTAL Given a transform ``t``, reshape to different batch shape
    of same number of elements.

    This is typically used to correct the transform parameters' shapes after
    reshaping the base distribution of a transformed distribution.

    :param t: A transform.
    :type t: ~torch.distributions.transforms.Transform
    :param tuple old_shape: The current shape of ``t``.
    :param tuple new_shape: A new shape.
    :returns: A transform with the same type but given new batch shape.
    :rtype: ~torch.distributions.transforms.Transform
    """
    raise NotImplementedError("reshape_transform_batch() does not suport {}".format(type(t)))


def _reshape_batch_univariate_transform(t, old_shape, new_shape):
    params = {name: getattr(t, name).expand(old_shape).reshape(new_shape)
              for name in UNIVARIATE_TRANSFORMS[type(t)]}
    params["cache_size"] = t._cache_size
    return type(t)(**params)


for _type in UNIVARIATE_TRANSFORMS:
    reshape_transform_batch.register(_type)(_reshape_batch_univariate_transform)


@reshape_transform_batch.register(dist.transforms.HaarTransform)
@reshape_transform_batch.register(dist.transforms.DiscreteCosineTransform)
def _(t, old_shape, new_shape):
    dim = t.event_dim
    assert len(old_shape) >= dim
    assert len(new_shape) >= dim
    if old_shape[-dim:] != new_shape[-dim:]:
        raise NotImplementedError
    return t
