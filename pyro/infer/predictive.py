# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, fields
from functools import reduce
from typing import Callable, List, Union

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.autoguide.initialization import InitMessenger, init_to_sample
from pyro.infer.importance import LogWeightsMixin
from pyro.infer.util import CloneMixin, plate_log_prob_sum
from pyro.poutine.trace_struct import Trace
from pyro.poutine.util import prune_subsample_sites


def _guess_max_plate_nesting(model, args, kwargs):
    """
    Guesses max_plate_nesting by running the model once
    without enumeration. This optimistically assumes static model
    structure.
    """
    with poutine.block():
        model_trace = poutine.trace(model).get_trace(*args, **kwargs)
    sites = [site for site in model_trace.nodes.values() if site["type"] == "sample"]

    dims = [
        frame.dim
        for site in sites
        for frame in site["cond_indep_stack"]
        if frame.vectorized
    ]
    max_plate_nesting = -min(dims) if dims else 0
    return max_plate_nesting


@dataclass(frozen=True, eq=False)
class _predictiveResults:
    """
    Return value of call to ``_predictive`` and ``_predictive_sequential``.
    """

    samples: dict
    trace: Union[Trace, List[Trace]]


def _predictive_sequential(
    model, posterior_samples, model_args, model_kwargs, num_samples, return_site_shapes
):
    collected_samples = []
    collected_trace = []
    samples = [
        {k: v[i] for k, v in posterior_samples.items()} for i in range(num_samples)
    ]
    for i in range(num_samples):
        trace = poutine.trace(poutine.condition(model, samples[i])).get_trace(
            *model_args, **model_kwargs
        )
        collected_trace.append(trace)
        collected_samples.append(
            {site: trace.nodes[site]["value"] for site in return_site_shapes}
        )

    return _predictiveResults(
        trace=collected_trace,
        samples={
            site: torch.stack([s[site] for s in collected_samples]).reshape(shape)
            for site, shape in return_site_shapes.items()
        },
    )


_predictive_vectorize_plate_name = "_num_predictive_samples"


def _predictive(
    model,
    posterior_samples,
    num_samples,
    return_sites=(),
    parallel=False,
    model_args=(),
    model_kwargs={},
    mask=True,
):
    model = torch.no_grad()(poutine.mask(model, mask=False) if mask else model)
    initailized_model = InitMessenger(init_to_sample)(model)
    max_plate_nesting = _guess_max_plate_nesting(
        initailized_model, model_args, model_kwargs
    )
    vectorize = pyro.plate(
        _predictive_vectorize_plate_name, num_samples, dim=-max_plate_nesting - 1
    )
    model_trace = prune_subsample_sites(
        poutine.trace(initailized_model).get_trace(*model_args, **model_kwargs)
    )
    reshaped_samples = {}

    for name, sample in posterior_samples.items():
        sample_shape = sample.shape[1:]
        sample = sample.reshape(
            (num_samples,)
            + (1,) * (max_plate_nesting - len(sample_shape))
            + sample_shape
        )
        reshaped_samples[name] = sample

    return_site_shapes = {}
    for site in model_trace.stochastic_nodes + model_trace.observation_nodes:
        append_ndim = max_plate_nesting - len(model_trace.nodes[site]["fn"].batch_shape)
        site_shape = (
            (num_samples,) + (1,) * append_ndim + model_trace.nodes[site]["value"].shape
        )
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
    if return_sites is not None and "_RETURN" in return_sites:
        value = model_trace.nodes["_RETURN"]["value"]
        shape = (num_samples,) + value.shape if torch.is_tensor(value) else None
        return_site_shapes["_RETURN"] = shape

    if not parallel:
        return _predictive_sequential(
            model,
            posterior_samples,
            model_args,
            model_kwargs,
            num_samples,
            return_site_shapes,
        )

    trace = poutine.trace(
        poutine.condition(vectorize(model), reshaped_samples)
    ).get_trace(*model_args, **model_kwargs)
    predictions = {}
    for site, shape in return_site_shapes.items():
        value = trace.nodes[site]["value"]
        if site == "_RETURN" and shape is None:
            predictions[site] = value
            continue
        if value.numel() < reduce((lambda x, y: x * y), shape):
            predictions[site] = value.expand(shape)
        else:
            predictions[site] = value.reshape(shape)

    return _predictiveResults(trace=trace, samples=predictions)


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

    def __init__(
        self,
        model,
        posterior_samples=None,
        guide=None,
        num_samples=None,
        return_sites=(),
        parallel=False,
    ):
        super().__init__()
        if posterior_samples is None:
            if num_samples is None:
                raise ValueError(
                    "Either posterior_samples or num_samples must be specified."
                )
            posterior_samples = {}

        for name, sample in posterior_samples.items():
            batch_size = sample.shape[0]
            if num_samples is None:
                num_samples = batch_size
            elif num_samples != batch_size:
                warnings.warn(
                    "Sample's leading dimension size {} is different from the "
                    "provided {} num_samples argument. Defaulting to {}.".format(
                        batch_size, num_samples, batch_size
                    ),
                    UserWarning,
                )
                num_samples = batch_size

        if num_samples is None:
            raise ValueError(
                "No sample sites in posterior samples to infer `num_samples`."
            )

        if guide is not None and posterior_samples:
            raise ValueError(
                "`posterior_samples` cannot be provided with the `guide` argument."
            )

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
            posterior_samples = _predictive(
                self.guide,
                posterior_samples,
                self.num_samples,
                return_sites=None,
                parallel=self.parallel,
                model_args=args,
                model_kwargs=kwargs,
            ).samples
        return _predictive(
            self.model,
            posterior_samples,
            self.num_samples,
            return_sites=return_sites,
            parallel=self.parallel,
            model_args=args,
            model_kwargs=kwargs,
        ).samples

    def get_samples(self, *args, **kwargs):
        warnings.warn(
            "The method `.get_samples` has been deprecated in favor of `.forward`.",
            DeprecationWarning,
        )
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
            posterior_samples = _predictive(
                self.guide,
                posterior_samples,
                self.num_samples,
                parallel=self.parallel,
                model_args=args,
                model_kwargs=kwargs,
            ).samples
        return _predictive(
            self.model,
            posterior_samples,
            self.num_samples,
            parallel=True,
            model_args=args,
            model_kwargs=kwargs,
        ).trace


@dataclass(frozen=True, eq=False)
class WeighedPredictiveResults(LogWeightsMixin, CloneMixin):
    """
    Return value of call to instance of :class:`WeighedPredictive`.
    """

    samples: Union[dict, tuple]
    log_weights: torch.Tensor
    guide_log_prob: torch.Tensor
    model_log_prob: torch.Tensor


class WeighedPredictive(Predictive):
    """
    Class used to construct a weighed predictive distribution that is based
    on the same initialization interface as :class:`Predictive`.

    The methods `.forward` and `.call` can be called with an additional keyword argument
    ``model_guide`` which is the model used to create and optimize the guide (if not
    provided ``model_guide`` defaults to ``self.model``), and they return both samples and log_weights.

    The weights are calculated as the per sample gap between the model_guide log-probability
    and the guide log-probability (a guide must always be provided).

    A typical use case would be based on a ``model`` :math:`p(x,z)=p(x|z)p(z)` and ``guide`` :math:`q(z)`
    that has already been fitted to the model given observations :math:`p(X_{obs},z)`, both of which
    are provided at itialization of :class:`WeighedPredictive` (same as you would do with :class:`Predictive`).
    When calling an instance of :class:`WeighedPredictive` we provide the model given observations :math:`p(X_{obs},z)`
    as the keyword argument ``model_guide``.
    The resulting output would be the usual samples :math:`p(x|z)q(z)` returned by :class:`Predictive`,
    along with per sample weights :math:`p(X_{obs},z)/q(z)`. The samples and weights can be fed into
    :any:`weighed_quantile` in order to obtain the true quantiles of the resulting distribution.

    Note that the ``model`` can be more elaborate with sample sites :math:`y` that are not observed
    and are not part of the guide, if the samples sites :math:`y` are sampled after the observations
    and the latent variables sampled by the guide, such that :math:`p(x,y,z)=p(y|x,z)p(x|z)p(z)` where
    each element in the product represents a set of ``pyro.sample`` statements.
    """

    def call(self, *args, **kwargs):
        """
        Method `.call` that is backwards compatible with the same method found in :class:`Predictive`
        but can be called with an additional keyword argument `model_guide`
        which is the model used to create and optimize the guide.

        Returns :class:`WeighedPredictiveResults` which has attributes ``.samples`` and per sample
        weights ``.log_weights``.
        """
        result = self.forward(*args, **kwargs)
        return WeighedPredictiveResults(
            samples=tuple(v for _, v in sorted(result.items())),
            log_weights=result.log_weights,
            guide_log_prob=result.guide_log_prob,
            model_log_prob=result.model_log_prob,
        )

    def forward(self, *args, **kwargs):
        """
        Method `.forward` that is backwards compatible with the same method found in :class:`Predictive`
        but can be called with an additional keyword argument `model_guide`
        which is the model used to create and optimize the guide.

        Returns :class:`WeighedPredictiveResults` which has attributes ``.samples`` and per sample
        weights ``.log_weights``.
        """
        model_guide = kwargs.pop("model_guide", self.model)
        return_sites = self.return_sites
        # return all sites by default if a guide is provided.
        return_sites = None if not return_sites else return_sites
        guide_predictive = _predictive(
            self.guide,
            self.posterior_samples,
            self.num_samples,
            return_sites=None,
            parallel=self.parallel,
            model_args=args,
            model_kwargs=kwargs,
            mask=False,
        )
        posterior_samples = guide_predictive.samples
        model_predictive = _predictive(
            model_guide,
            posterior_samples,
            self.num_samples,
            return_sites=return_sites,
            parallel=self.parallel,
            model_args=args,
            model_kwargs=kwargs,
            mask=False,
        )
        if not isinstance(guide_predictive.trace, list):
            guide_trace = prune_subsample_sites(guide_predictive.trace)
            model_trace = prune_subsample_sites(model_predictive.trace)
            guide_trace.compute_score_parts()
            model_trace.compute_log_prob()
            guide_trace.pack_tensors()
            model_trace.pack_tensors(guide_trace.plate_to_symbol)
            plate_symbol = guide_trace.plate_to_symbol[_predictive_vectorize_plate_name]
            guide_log_prob = plate_log_prob_sum(guide_trace, plate_symbol)
            model_log_prob = plate_log_prob_sum(model_trace, plate_symbol)
        else:
            guide_log_prob = torch.Tensor(
                [
                    trace_element.log_prob_sum()
                    for trace_element in guide_predictive.trace
                ]
            )
            model_log_prob = torch.Tensor(
                [
                    trace_element.log_prob_sum()
                    for trace_element in model_predictive.trace
                ]
            )
        return WeighedPredictiveResults(
            samples=(
                _predictive(
                    self.model,
                    posterior_samples,
                    self.num_samples,
                    return_sites=return_sites,
                    parallel=self.parallel,
                    model_args=args,
                    model_kwargs=kwargs,
                ).samples
                if model_guide is not self.model
                else model_predictive.samples
            ),
            log_weights=model_log_prob - guide_log_prob,
            guide_log_prob=guide_log_prob,
            model_log_prob=model_log_prob,
        )


class MHResampler(torch.nn.Module):
    r"""
    Resampler for weighed samples that generates equally weighed samples from the distribution
    specified by the weighed samples ``sampler``.

    The resampling is based on the Metropolis-Hastings algorithm.
    Given an initial sample :math:`x` subsequent samples are generated by:

    -   Sampling from the ``guide`` a new sample candidate :math:`x'` with probability :math:`g(x')`.
    -   Calculate an acceptance probability
        :math:`A(x', x) = \min\left(1, \frac{P(x')}{P(x)} \frac{g(x)}{g(x')}\right)`
        with :math:`P` being the ``model``.
    -   With probability :math:`A(x', x)` accept the new sample candidate :math:`x'`
        as the next sample, otherwise set the current sample :math:`x` as the next sample.

    The above is the Metropolis-Hastings algorithm with the new sample candidate
    proposal distribution being equal to the ``guide`` and independent of the
    current sample such that :math:`g(x')=g(x' \mid x)`.

    :param callable sampler: When called returns :class:`WeighedPredictiveResults`.
    :param slice source_samples_slice: Select source samples for storage (default is `slice(0)`, i.e. none).
    :param slice stored_samples_slice: Select output samples for storage (default is `slice(0)`, i.e. none).

    The typical use case of :class:`MHResampler` would be to convert weighed samples
    generated by :class:`WeighedPredictive` into equally weighed samples from the target distribution.
    Each time an instance of :class:`MHResampler` is called it returns a new set of samples, with the
    samples generated by the first call being distributed according to the ``guide``, and with each
    subsequent call the distribution of the samples becomes closer to that of the posterior predictive
    disdtribution. It might take some experimentation in order to find out in each case how many times one would
    need to call an instance of :class:`MHResampler` in order to be close enough to the posterior
    predictive distribution.

    Example::

        def model():
            ...

        def guide():
            ...

        def conditioned_model():
            ...

        # Fit guide
        elbo = Trace_ELBO(num_particles=100, vectorize_particles=True)
        svi = SVI(conditioned_model, guide, optim.Adam(dict(lr=3.0)), elbo)
        for i in range(num_svi_steps):
            svi.step()

        # Create callable that returns weighed samples
        posterior_predictive = WeighedPredictive(model,
                                                 guide=guide,
                                                 num_samples=num_samples,
                                                 parallel=parallel,
                                                 return_sites=["_RETURN"])

        prob = 0.95

        weighed_samples = posterior_predictive(model_guide=conditioned_model)
        # Calculate quantile directly from weighed samples
        weighed_samples_quantile = weighed_quantile(weighed_samples.samples['_RETURN'],
                                                    [prob],
                                                    weighed_samples.log_weights)[0]

        resampler = MHResampler(posterior_predictive)
        num_mh_steps = 10
        for mh_step_count in range(num_mh_steps):
            resampled_weighed_samples = resampler(model_guide=conditioned_model)
        # Calculate quantile from resampled weighed samples (samples are equally weighed)
        resampled_weighed_samples_quantile = quantile(resampled_weighed_samples.samples[`_RETURN`],
                                                      [prob])[0]

        # Quantiles calculated using both methods should be identical
        assert_close(weighed_samples_quantile, resampled_weighed_samples_quantile, rtol=0.01)

    .. _mhsampler-behavior:

    **Notes on Sampler Behavior:**

    -   In case the ``guide`` perfectly tracks the ``model`` this sampler will do nothing
        as the acceptance probability :math:`A(x', x)` will always be one.
    -   Furtheremore, if the guide is approximately separable, i.e. :math:`g(z_A, z_B) \approx g_A(z_A) g_B(z_B)`,
        with :math:`g_A(z_A)` pefectly tracking the ``model`` and :math:`g_B(z_B)` poorly tracking the ``model``,
        quantiles of :math:`z_A` calculated from samples taken from :class:`MHResampler`, will have much lower
        variance then quantiles of :math:`z_A` calculated by using :any:`weighed_quantile`, as the effective sample size
        of the calculation using :any:`weighed_quantile` will be low due to :math:`g_B(z_B)` poorly tracking
        the ``model``, whereas when using :class:`MHResampler` the poor ``model`` tracking of :math:`g_B(z_B)` has
        negligible affect on the effective sample size of :math:`z_A` samples.
    """

    def __init__(
        self,
        sampler: Callable,
        source_samples_slice: slice = slice(0),
        stored_samples_slice: slice = slice(0),
    ):
        super().__init__()
        self.sampler = sampler
        self.samples = None
        self.transition_count = torch.tensor(0, dtype=torch.long)
        self.source_samples = []
        self.source_samples_slice = source_samples_slice
        self.stored_samples = []
        self.stored_samples_slice = stored_samples_slice

    def forward(self, *args, **kwargs):
        """
        Perform single resampling step.
        Returns :class:`WeighedPredictiveResults`
        """
        with torch.no_grad():
            new_samples = self.sampler(*args, **kwargs)
            # Store samples
            self.source_samples.append(new_samples)
            self.source_samples = self.source_samples[self.source_samples_slice]
            if self.samples is None:
                # First set of samples
                self.samples = new_samples.clone()
                self.transition_count = torch.zeros_like(
                    new_samples.log_weights, dtype=torch.long
                )
            else:
                # Apply Metropolis-Hastings algorithm
                prob = torch.clamp(
                    new_samples.log_weights - self.samples.log_weights, max=0.0
                ).exp()
                idx = torch.rand(*prob.shape) <= prob
                self.transition_count[idx] += 1
                for field_desc in fields(self.samples):
                    field, new_field = getattr(self.samples, field_desc.name), getattr(
                        new_samples, field_desc.name
                    )
                    if isinstance(field, dict):
                        for key in field:
                            field[key][idx] = new_field[key][idx]
                    else:
                        field[idx] = new_field[idx]
        self.stored_samples.append(self.samples.clone())
        self.stored_samples = self.stored_samples[self.stored_samples_slice]
        return self.samples

    def get_min_sample_transition_count(self):
        """
        Return transition count of sample with minimal amount of transitions.
        """
        return self.transition_count.min()

    def get_total_transition_count(self):
        """
        Return total number of transitions.
        """
        return self.transition_count.sum()

    def get_source_samples(self):
        """
        Return source samples that were the input to the Metropolis-Hastings algorithm.
        """
        return self.get_samples(self.source_samples)

    def get_stored_samples(self):
        """
        Return stored samples that were the output of the Metropolis-Hastings algorithm.
        """
        return self.get_samples(self.stored_samples)

    def get_samples(self, samples):
        """
        Return samples that were sampled during execution of the Metropolis-Hastings algorithm.
        """
        retval = dict()
        for field_desc in fields(self.samples):
            field_name, value = field_desc.name, getattr(self.samples, field_desc.name)
            if isinstance(value, dict):
                retval[field_name] = dict()
                for key in value:
                    retval[field_name][key] = torch.cat(
                        [getattr(sample, field_name)[key] for sample in samples]
                    )
            else:
                retval[field_name] = torch.cat(
                    [getattr(sample, field_name) for sample in samples]
                )
        return self.samples.__class__(**retval)
