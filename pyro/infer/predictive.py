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


def predictive(model, posterior_samples, *args, num_samples=None, return_sites=None,
               return_trace=False, parallel=False, **kwargs):
    """
    Run model by sampling latent parameters from `posterior_samples`, and return
    values at sample sites from the forward run. By default, only sites not contained in
    `posterior_samples` are returned. This can be modified by changing the `return_sites`
    keyword argument.

    .. note:: Using empty dict `posterior_samples={}` for prior predictive or getting posterior
        samples from `guide` in SVI. In those predictive, the number of samples `num_samples`
        should be specified.

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
    # num_samples = kwargs.pop('num_samples', None)
    # return_sites = kwargs.pop('return_sites', None)
    # return_trace = kwargs.pop('return_trace', False)
    # parallel = kwargs.pop('parallel', False)

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
        if return_sites:
            if site in return_sites:
                return_site_shapes[site] = site_shape
        else:
            if site not in reshaped_samples:
                return_site_shapes[site] = site_shape

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
        if value.numel() < reduce((lambda x, y: x * y), shape):
            predictions[site] = value.expand(shape)
        else:
            predictions[site] = value.reshape(shape)

    return predictions
