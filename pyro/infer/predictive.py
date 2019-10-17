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
                           num_samples, sample_sites, return_trace=False):
    collected = []
    samples = [{k: v[i] for k, v in posterior_samples.items()} for i in range(num_samples)]
    for i in range(num_samples):
        trace = poutine.trace(poutine.condition(model, samples[i])).get_trace(*model_args, **model_kwargs)
        if return_trace:
            collected.append(trace)
        else:
            collected.append({site: trace.nodes[site]['value'] for site in sample_sites})

    return collected if return_trace else {site: torch.stack([s[site] for s in collected])
                                           for site in sample_sites}


def _predictive(model, posterior_samples, *args, **kwargs):
    num_samples = kwargs.pop('num_samples', None)
    return_sites = kwargs.pop('return_sites', None)
    return_trace = kwargs.pop('return_trace', False)
    parallel = kwargs.pop('parallel', False)

    max_plate_nesting = _guess_max_plate_nesting(model, args, kwargs)
    model_trace = prune_subsample_sites(poutine.trace(model).get_trace(*args, **kwargs))
    reshaped_samples = {}

    for name, sample in posterior_samples.items():

        batch_size, sample_shape = sample.shape[0], sample.shape[1:]

        if num_samples is None:
            num_samples = batch_size

        elif num_samples != batch_size:
            warnings.warn("Sample's leading dimension size {} is different from the "
                          "provided {} num_samples argument. Defaulting to {}."
                          .format(batch_size, num_samples, batch_size), UserWarning)
            num_samples = batch_size

        sample = sample.reshape((num_samples,) + (1,) * (max_plate_nesting - len(sample_shape)) + sample_shape)
        reshaped_samples[name] = sample

    if num_samples is None:
        raise ValueError("No sample sites in model to infer `num_samples`.")

    return_site_shapes = {}
    for site in model_trace.stochastic_nodes + model_trace.observation_nodes:
        site_shape = (num_samples,) + model_trace.nodes[site]['value'].shape
        if isinstance(return_sites, (list, tuple, set)):
            if site in return_sites:
                return_site_shapes[site] = site_shape
        else:
            if (return_sites is not None) or (site not in reshaped_samples):
                return_site_shapes[site] = site_shape

    # handle _RETURN site
    if isinstance(return_sites, (list, tuple, set)) and '_RETURN' in return_sites:
        value = model_trace.nodes['_RETURN']['value']
        shape = (num_samples,) + value.shape if torch.is_tensor(value) else None
        return_site_shapes['_RETURN'] = shape

    if not parallel:
        return _predictive_sequential(model, posterior_samples, args, kwargs, num_samples,
                                      return_site_shapes.keys(), return_trace)

    def _vectorized_fn(fn):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        sampling from the posterior predictive.

        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            with pyro.plate("_num_predictive_samples", num_samples, dim=-max_plate_nesting-1):
                return fn(*args, **kwargs)

        return wrapped_fn

    trace = poutine.trace(poutine.condition(_vectorized_fn(model), reshaped_samples))\
        .get_trace(*args, **kwargs)

    if return_trace:
        return trace

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


def predictive(model, posterior_samples, *args, **kwargs):
    """
    Run model by sampling latent parameters from `posterior_samples`, and return
    values at sample sites from the forward run. By default, only sample sites not contained in
    `posterior_samples` are returned. This can be modified by changing the `return_sites`
    keyword argument.

    .. warning::
        The interface for the `predictive` class is experimental, and
        might change in the future.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param args: model arguments.
    :param kwargs: model kwargs; and other keyword arguments (see below).

    :Keyword Arguments:
        * **num_samples** (``int``) - number of samples to draw from the predictive distribution.
          This argument has no effect if ``posterior_samples`` is non-empty, in which case, the
          leading dimension size of samples in ``posterior_samples`` is used.
        * **guide** (``callable``) - guide to get posterior samples of sites not present
          in `posterior_samples`.
        * **return_sites** (``list``) - sites to return; by default only sample sites not present
          in `posterior_samples` are returned.
        * **return_trace** (``bool``) - whether to return the full trace. Note that this is
          vectorized over `num_samples`.
        * **parallel** (``bool``) - predict in parallel by wrapping the existing model
          in an outermost `plate` messenger. Note that this requires that the model has
          all batch dims correctly annotated via :class:`~pyro.plate`. Default is `False`.

    :return: dict of samples from the predictive distribution, or a single vectorized
        `trace` (if `return_trace=True`).
    """
    guide = kwargs.pop('guide', None)
    return_trace = kwargs.pop('return_trace', False)
    return_sites = kwargs.pop('return_sites', None)
    if return_sites is not None:
        assert isinstance(return_sites, (list, tuple, set))
    if guide is not None:
        # use return_sites='' as a special signal to return all sites
        posterior_samples = _predictive(guide, posterior_samples, *args, return_trace=False, return_sites='', **kwargs)
    return _predictive(model, posterior_samples, *args, return_trace=return_trace, return_sites=return_sites, **kwargs)
