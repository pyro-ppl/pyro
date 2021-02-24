# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from typing import Callable, Dict
import warnings

import torch

import pyro
import pyro.poutine as poutine
from pyro.poutine.util import prune_subsample_sites


def _guess_max_plate_nesting(model, args, kwargs):
    """
    Guesses max_plate_nesting by running the model once
    without enumeration. This optimistically assumes static model
    structure.
    """
    with poutine.block():
        model_trace = poutine.trace(model).get_trace(*args, **kwargs)
    sites = [site for site in model_trace.nodes.values()
             if site["type"] == "sample"]

    dims = [frame.dim
            for site in sites
            for frame in site["cond_indep_stack"]
            if frame.vectorized]
    max_plate_nesting = -min(dims) if dims else 0
    return max_plate_nesting


def _predictive_sequential(model, posterior_samples, model_args, model_kwargs,
                           num_samples, return_site_shapes, return_trace=False,
                           collect_fn=None):
    collected = []
    if collect_fn is None:
        def collect_fn(msg):
            return msg["value"]
    samples = [{k: v[i] for k, v in posterior_samples.items()} for i in range(num_samples)]
    for i in range(num_samples):
        trace = poutine.trace(poutine.condition(model, samples[i])).get_trace(*model_args, **model_kwargs)
        if return_trace:
            collected.append(trace)
        else:
            collected.append({site: collect_fn(trace.nodes[site]) for site in return_site_shapes})

    if return_trace:
        return collected
    else:
        return {site: torch.stack([s[site] for s in collected]).reshape(shape)
                for site, shape in return_site_shapes.items()}


def _predictive(model, posterior_samples, num_samples, return_sites=(),
                return_trace=False, parallel=False, model_args=(), model_kwargs={}):
    model = torch.no_grad()(poutine.mask(model, mask=False))
    max_plate_nesting = _guess_max_plate_nesting(model, model_args, model_kwargs)
    vectorize = pyro.plate("_num_predictive_samples", num_samples, dim=-max_plate_nesting-1)
    model_trace = prune_subsample_sites(poutine.trace(model).get_trace(*model_args, **model_kwargs))
    reshaped_samples = {}

    for name, sample in posterior_samples.items():
        sample_shape = sample.shape[1:]
        sample = sample.reshape((num_samples,) + (1,) * (max_plate_nesting - len(sample_shape)) + sample_shape)
        reshaped_samples[name] = sample

    if return_trace:
        trace = poutine.trace(poutine.condition(vectorize(model), reshaped_samples))\
            .get_trace(*model_args, **model_kwargs)
        return trace

    return_site_shapes = {}
    for site in model_trace.stochastic_nodes + model_trace.observation_nodes:
        append_ndim = max_plate_nesting - len(model_trace.nodes[site]["fn"].batch_shape)
        site_shape = (num_samples,) + (1,) * append_ndim + model_trace.nodes[site]['value'].shape
        # non-empty return-sites
        if return_sites:
            if site in return_sites:
                return_site_shapes[site] = site_shape
        # special case (for guides): include all sites
        elif return_sites is None:
            return_site_shapes[site] = site_shape
        # default case: return sites = ()
        # include all sites not in posterior samples
        elif site not in posterior_samples:
            return_site_shapes[site] = site_shape

    # handle _RETURN site
    if return_sites is not None and '_RETURN' in return_sites:
        value = model_trace.nodes['_RETURN']['value']
        shape = (num_samples,) + value.shape if torch.is_tensor(value) else None
        return_site_shapes['_RETURN'] = shape

    if not parallel:
        return _predictive_sequential(model, posterior_samples, model_args, model_kwargs, num_samples,
                                      return_site_shapes, return_trace=False)

    trace = poutine.trace(poutine.condition(vectorize(model), reshaped_samples))\
        .get_trace(*model_args, **model_kwargs)
    predictions = {}
    for site, shape in return_site_shapes.items():
        value = trace.nodes[site]['value']
        if site == '_RETURN' and shape is None:
            predictions[site] = value
            continue
        if value.numel() < reduce((lambda x, y: x * y), shape):
            predictions[site] = value.expand(shape)
        else:
            predictions[site] = value.reshape(shape)

    return predictions


def _validate_predictive_inputs(posterior_samples, num_samples, guide, return_sites):
    if posterior_samples is None:
        if num_samples is None:
            raise ValueError("Either posterior_samples or num_samples must be specified.")
        posterior_samples = {}

    for name, sample in posterior_samples.items():
        batch_size = sample.shape[0]
        if num_samples is None:
            num_samples = batch_size
        elif num_samples != batch_size:
            warnings.warn("Sample's leading dimension size {} is different from the "
                          "provided {} num_samples argument. Defaulting to {}."
                          .format(batch_size, num_samples, batch_size), UserWarning)
            num_samples = batch_size

    if num_samples is None:
        raise ValueError("No sample sites in posterior samples to infer `num_samples`.")

    if guide is not None and posterior_samples:
        raise ValueError("`posterior_samples` cannot be provided with the `guide` argument.")

    if return_sites is not None:
        assert isinstance(return_sites, (list, tuple, set))

    return posterior_samples, num_samples


class Predictive(torch.nn.Module):
    """
    EXPERIMENTAL class used to construct predictive distribution. The predictive
    distribution is obtained by running the `model` conditioned on latent samples
    from `posterior_samples`. If a `guide` is provided, then posterior samples
    from all the latent sites are also returned.

    .. warning::
        The interface for the :class:`Predictive` class is experimental, and
        might change in the future.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param callable guide: optional guide to get posterior samples of sites not present
        in `posterior_samples`.
    :param int num_samples: number of samples to draw from the predictive distribution.
        This argument has no effect if ``posterior_samples`` is non-empty, in which case,
        the leading dimension size of samples in ``posterior_samples`` is used.
    :param return_sites: sites to return; by default only sample sites not present
        in `posterior_samples` are returned.
    :type return_sites: list, tuple, or set
    :param bool parallel: predict in parallel by wrapping the existing model
        in an outermost `plate` messenger. Note that this requires that the model has
        all batch dims correctly annotated via :class:`~pyro.plate`. Default is `False`.
    """
    def __init__(self, model, posterior_samples=None, guide=None, num_samples=None,
                 return_sites=(), parallel=False):
        super().__init__()
        posterior_samples, num_samples = _validate_predictive_inputs(
            posterior_samples, num_samples, guide, return_sites)

        self.model = model
        self.posterior_samples = {} if posterior_samples is None else posterior_samples
        self.num_samples = num_samples
        self.guide = guide
        self.return_sites = return_sites
        self.parallel = parallel

    def call(self, *args, **kwargs):
        """
        Method that calls :meth:`forward` and returns parameter values of the
        guide as a `tuple` instead of a `dict`, which is a requirement for
        JIT tracing. Unlike :meth:`forward`, this method can be traced by
        :func:`torch.jit.trace_module`.

        .. warning::
            This method may be removed once PyTorch JIT tracer starts accepting
            `dict` as valid return types. See
            `issue <https://github.com/pytorch/pytorch/issues/27743>`_.
        """
        result = self.forward(*args, **kwargs)
        return tuple(v for _, v in sorted(result.items()))

    def forward(self, *args, **kwargs):
        """
        Returns dict of samples from the predictive distribution. By default, only sample sites not
        contained in `posterior_samples` are returned. This can be modified by changing the
        `return_sites` keyword argument of this :class:`Predictive` instance.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__` as in
            ``Predictive(model)(*args, **kwargs)``.

        :param args: model arguments.
        :param kwargs: model keyword arguments.
        """
        posterior_samples = self.posterior_samples
        return_sites = self.return_sites
        if self.guide is not None:
            # return all sites by default if a guide is provided.
            return_sites = None if not return_sites else return_sites
            posterior_samples = _predictive(self.guide, posterior_samples, self.num_samples, return_sites=None,
                                            parallel=self.parallel, model_args=args, model_kwargs=kwargs)
        return _predictive(self.model, posterior_samples, self.num_samples, return_sites=return_sites,
                           parallel=self.parallel, model_args=args, model_kwargs=kwargs)

    def get_samples(self, *args, **kwargs):
        warnings.warn("The method `.get_samples` has been deprecated in favor of `.forward`.",
                      DeprecationWarning)
        return self.forward(*args, **kwargs)

    def get_vectorized_trace(self, *args, **kwargs):
        """
        Returns a single vectorized `trace` from the predictive distribution. Note that this
        requires that the model has all batch dims correctly annotated via :class:`~pyro.plate`.

        :param args: model arguments.
        :param kwargs: model keyword arguments.
        """
        posterior_samples = self.posterior_samples
        if self.guide is not None:
            posterior_samples = _predictive(self.guide, posterior_samples, self.num_samples,
                                            parallel=self.parallel, model_args=args, model_kwargs=kwargs)
        return _predictive(self.model, posterior_samples, self.num_samples,
                           return_trace=True, model_args=args, model_kwargs=kwargs)


def log_likelihood(
    model: Callable,
    posterior_samples: Dict[str, torch.Tensor] = None,
    guide: Callable = None,
    num_samples: int = None,
    parallel: bool = True,
    obs_shapes: Dict[str, torch.Size] = None,
) -> Callable:
    """EXPERIMENTAL utility to calculate the log likelihood of the data for a
    model with observed sites, given posterior samples (or a guide that can
    generate them).

    :param model: Model function containing pyro primitives. Must have observed
        sites conditioned on data.
    :type model: Callable
    :param posterior_samples: Dictionary of posterior samples, defaults to None
    :type posterior_samples: Dict[str, torch.Tensor], optional
    :param guide: Guide function, defaults to None
    :type guide: Callable, optional
    :param num_samples: Number of posterior samples to evaluate, defaults to None
    :type num_samples: int, optional
    :param parallel: Whether to use vectorization, defaults to True
    :type parallel: bool, optional
    :param obs_shapes: A dictionary mapping the observed site names to the
        expected batch size of the returned tensor. If this is not supplied, it
        will be automatically generated by running an extra trace of the model.
    :type obs_shapes: Dict[str, torch.Size], optional
    :rtype: Callable

    The API parallels that of the :class:`Predictive` interface (assume `model`
    and `guide` have already been defined elsewhere in the below example):

    ```
    x = torch.randn(10)
    y = x + 0.05 * torch.randn(10)
    cond_model = pyro.condition(model, data={"y": y})
    ll_vals = log_likelihood(cond_model, guide=guide, num_samples)(x)
    ```
    """

    posterior_samples, num_samples = _validate_predictive_inputs(
        posterior_samples, num_samples, guide, return_sites=None)

    # define the function to return the likelihood values
    def log_like_helper(*args, **kwargs) -> torch.Tensor:
        if obs_shapes is None:
            # trace model once to get observed nodes
            trace = pyro.poutine.trace(model).get_trace(*args, **kwargs)
            observed = {
                name: site["fn"].batch_shape
                for name, site in trace.nodes.items()
                if site["type"] == "sample" and site["is_observed"]
            }
        else:
            observed = obs_shapes.copy()
        if len(observed) == 0:
            raise RuntimeError("No observed sites in the model")
        # use vectorized trace if parallelizable
        if parallel:
            log_like = dict()
            # now trace it and extract the likelihood from observed sites
            predictive = Predictive(
                model, posterior_samples, guide, num_samples, (), parallel)
            trace = predictive.get_vectorized_trace(*args, **kwargs)
            for obs_name in observed.keys():
                site = trace.nodes[obs_name]
                log_like[obs_name] = site["fn"].log_prob(site["value"])
            return log_like
        # iterate over samples from posterior if model can't be vectorized
        else:
            # use guide to get posterior samples if provided
            if guide is not None:
                _posterior_samples = _predictive(
                    guide, {}, num_samples, model_args=args,
                    model_kwargs=kwargs, parallel=False)
            else:
                _posterior_samples = posterior_samples
            # use _predictive_sequential to collect log likelihood values
            site_shapes = {
                name: (num_samples,) + shape for name, shape in observed.items()
            }
            log_like = _predictive_sequential(
                model, _posterior_samples, args, kwargs, num_samples, site_shapes,
                collect_fn=lambda msg: msg["fn"].log_prob(msg["value"]))
            return log_like
    return log_like_helper
