# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
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
                           num_samples, return_site_shapes, return_trace=False):
    collected = []
    samples = [{k: v[i] for k, v in posterior_samples.items()} for i in range(num_samples)]
    for i in range(num_samples):
        trace = poutine.trace(poutine.condition(model, samples[i])).get_trace(*model_args, **model_kwargs)
        if return_trace:
            collected.append(trace)
        else:
            collected.append({site: trace.nodes[site]['value'] for site in return_site_shapes})

    if return_trace:
        return collected
    else:
        return {site: torch.stack([s[site] for s in collected]).reshape(shape)
                for site, shape in return_site_shapes.items()}


def _predictive(model, posterior_samples, num_samples, return_sites=(),
                return_trace=False, parallel=False, model_args=(), model_kwargs={}):
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
