# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import operator
import re
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import reduce
from timeit import default_timer

import torch
from torch.distributions import biject_to, constraints
from torch.distributions.utils import lazy_property

import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.distributions.transforms import HaarTransform
from pyro.infer import MCMC, NUTS, SVI, JitTrace_ELBO, SMCFilter, Trace_ELBO, infer_discrete
from pyro.infer.autoguide import (AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoNormal, init_to_generated,
                                  init_to_value)
from pyro.infer.mcmc import ArrowheadMassMatrix
from pyro.infer.reparam import HaarReparam, SplitReparam
from pyro.infer.smcfilter import SMCFailed
from pyro.infer.util import is_validation_enabled
from pyro.optim import ClippedAdam
from pyro.poutine.util import site_is_factor, site_is_subsample
from pyro.util import warn_if_nan

from .distributions import set_approx_log_prob_tol, set_approx_sample_thresh, set_relaxed_distributions
from .util import align_samples, cat2, clamp, quantize, quantize_enumerate

logger = logging.getLogger(__name__)


def _require_double_precision():
    if torch.get_default_dtype() != torch.float64:
        warnings.warn("CompartmentalModel is unstable for dtypes less than torch.float64; "
                      "try torch.set_default_dtype(torch.float64)",
                      RuntimeWarning)


@contextmanager
def _disallow_latent_variables(section_name):
    if not is_validation_enabled():
        yield
        return

    with poutine.trace() as tr:
        yield
    for name, site in tr.trace.nodes.items():
        if site["type"] == "sample" and not site["is_observed"]:
            raise NotImplementedError("{} contained latent variable {}"
                                      .format(section_name, name))


class CompartmentalModel(ABC):
    """
    Abstract base class for discrete-time discrete-value stochastic
    compartmental models.

    Derived classes must implement methods :meth:`initialize` and
    :meth:`transition`. Derived classes may optionally implement
    :meth:`global_model`, :meth:`compute_flows`, and :meth:`heuristic`.

    Example usage::

        # First implement a concrete derived class.
        class MyModel(CompartmentalModel):
            def __init__(self, ...): ...
            def global_model(self): ...
            def initialize(self, params): ...
            def transition(self, params, state, t): ...

        # Run inference to fit the model to data.
        model = MyModel(...)
        model.fit_svi(num_samples=100)  # or .fit_mcmc(...)
        R0 = model.samples["R0"]  # An example parameter.
        print("R0 = {:0.3g} \u00B1 {:0.3g}".format(R0.mean(), R0.std()))

        # Predict latent variables.
        samples = model.predict()

        # Forecast forward.
        samples = model.predict(forecast=30)

        # You can assess future interventions (applied after ``duration``) by
        # storing them as attributes that are read by your derived methods.
        model.my_intervention = False
        samples1 = model.predict(forecast=30)
        model.my_intervention = True
        samples2 = model.predict(forecast=30)
        effect = samples2["my_result"].mean() - samples1["my_result"].mean()
        print("average effect = {:0.3g}".format(effect))

    An example workflow is to use cheaper approximate inference while finding
    good model structure and priors, then move to more accurate but more
    expensive inference once the model is plausible.

    1.  Start with ``.fit_svi(guide_rank=0, num_steps=2000)`` for cheap
        inference while you search for a good model.
    2.  Additionally infer long-range correlations by moving to a low-rank
        multivariate normal guide via ``.fit_svi(guide_rank=None,
        num_steps=5000)``.
    3.  Optionally additionally infer non-Gaussian posterior by moving to the
        more expensive (but still approximate via moment matching)
        ``.fit_mcmc(num_quant_bins=1, num_samples=10000, num_chains=2)``.
    4.  Optionally improve fit around small counts by moving the the more
        expensive enumeration-based algorithm ``.fit_mcmc(num_quant_bins=4,
        num_samples=10000, num_chains=2)`` (GPU recommended).

    :ivar dict samples: Dictionary of posterior samples.
    :param list compartments: A list of strings of compartment names.
    :param int duration: The number of discrete time steps in this model.
    :param population: Either the total population of a single-region model or
        a tensor of each region's population in a regional model.
    :type population: int or torch.Tensor
    :param tuple approximate: Names of compartments for which pointwise
        approximations should be provided in :meth:`transition`, e.g. if you
        specify ``approximate=("I")`` then the ``state["I_approx"]`` will be a
        continuous-valued non-enumerated point estimate of ``state["I"]``.
        Approximations are useful to reduce computational cost. Approximations
        are continuous-valued with support ``(-0.5, population + 0.5)``.
    """

    def __init__(self, compartments, duration, population, *,
                 approximate=()):
        super().__init__()

        assert isinstance(duration, int)
        assert duration >= 1
        self.duration = duration

        if isinstance(population, torch.Tensor):
            assert population.dim() == 1
            assert (population >= 1).all()
            self.is_regional = True
            self.max_plate_nesting = 2  # [time, region]
        else:
            assert isinstance(population, int)
            assert population >= 2
            self.is_regional = False
            self.max_plate_nesting = 1  # [time]
        self.population = population

        compartments = tuple(compartments)
        assert all(isinstance(name, str) for name in compartments)
        assert len(compartments) == len(set(compartments))
        self.compartments = compartments

        assert isinstance(approximate, tuple)
        assert all(name in compartments for name in approximate)
        self.approximate = approximate

        # Inference state.
        self.samples = {}
        self._clear_plates()

    @property
    def time_plate(self):
        """
        A ``pyro.plate`` for the time dimension.
        """
        if self._time_plate is None:
            self._time_plate = pyro.plate("time", self.duration,
                                          dim=-2 if self.is_regional else -1)
        return self._time_plate

    @property
    def region_plate(self):
        """
        Either a ``pyro.plate`` or a trivial ``ExitStack`` depending on whether
        this model ``.is_regional``.
        """
        if self._region_plate is None:
            if self.is_regional:
                self._region_plate = pyro.plate("region", len(self.population), dim=-1)
            else:
                self._region_plate = ExitStack()  # Trivial context manager.
        return self._region_plate

    def _clear_plates(self):
        self._time_plate = None
        self._region_plate = None

    @lazy_property
    def full_mass(self):
        """
        A list of a single tuple of the names of global random variables.
        """
        with torch.no_grad(), poutine.block(), poutine.trace() as tr:
            self.global_model()
        return [tuple(name for name, site in tr.trace.iter_stochastic_nodes()
                      if not site_is_subsample(site))]

    @lazy_property
    def series(self):
        """
        A frozenset of names of sample sites that are sampled each time step.
        """
        # Trace a simple invocation of .transition().
        with torch.no_grad(), poutine.block():
            params = self.global_model()
            prev = self.initialize(params)
            for name in self.approximate:
                prev[name + "_approx"] = prev[name]
            curr = prev.copy()
            with poutine.trace() as tr:
                self.transition(params, curr, 0)
        return frozenset(re.match("(.*)_0", name).group(1)
                         for name, site in tr.trace.nodes.items()
                         if site["type"] == "sample"
                         if not site_is_subsample(site))

    # Overridable attributes and methods ########################################

    def global_model(self):
        """
        Samples and returns any global parameters.

        :returns: An arbitrary object of parameters (e.g. ``None`` or a tuple).
        """
        return None

    # TODO Allow stochastic initialization.
    @abstractmethod
    def initialize(self, params):
        """
        Returns initial counts in each compartment.

        :param params: The global params returned by :meth:`global_model`.
        :returns: A dict mapping compartment name to initial value.
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def transition(self, params, state, t):
        """
        Forward generative process for dynamics.

        This inputs a current ``state`` and stochastically updates that
        state in-place.

        Note that this method is called under multiple different
        interpretations, including batched and vectorized interpretations.
        During :meth:`generate` this is called to generate a single sample.
        During :meth:`heuristic` this is called to generate a batch of samples
        for SMC.  During :meth:`fit_mcmc` this is called both in vectorized form
        (vectorizing over time) and in sequential form (for a single time
        step); both forms enumerate over discrete latent variables.  During
        :meth:`predict` this is called to forecast a batch of samples,
        conditioned on posterior samples for the time interval
        ``[0:duration]``.

        :param params: The global params returned by :meth:`global_model`.
        :param dict state: A dictionary mapping compartment name to current
            tensor value. This should be updated in-place.
        :param t: A time-like index. During inference ``t`` may be either a
            slice (for vectorized inference) or an integer time index. During
            prediction ``t`` will be integer time index.
        :type t: int or slice
        """
        raise NotImplementedError

    def finalize(self, params, prev, curr):
        """
        Optional method for likelihoods that depend on entire time series.

        This should be used only for non-factorizable likelihoods that couple
        states across time. Factorizable likelihoods should instead be added to
        the :meth:`transition` method, thereby enabling their use in
        :meth:`heuristic` initialization. Since this method is called only
        after the last time step, it is not used in :meth:`heuristic`
        initialization.

        .. warning:: This currently does not support latent variables.

        :param params: The global params returned by :meth:`global_model`.
        :param dict prev:
        :param dict curr: Dictionaries mapping compartment name to tensor of
            entire time series. These two parameters are offset by 1 step,
            thereby making it easy to compute time series of fluxes. For
            quantized inference, this uses the approximate point estimates, so
            users must request any needed time series in :meth:`__init__`, e.g.
            by calling ``super().__init__(..., approximate=("I", "E"))`` if
            likelihood depends on the ``I`` and ``E`` time series.
        """
        pass

    def compute_flows(self, prev, curr, t):
        """
        Computes flows between compartments, given compartment populations
        before and after time step t.

        The default implementation assumes sequential flows terminating in an
        implicit compartment named "R". For example if::

            compartment_names = ("S", "E", "I")

        the default implementation computes at time step ``t = 9``::

            flows["S2E_9"] = prev["S"] - curr["S"]
            flows["E2I_9"] = prev["E"] - curr["E"] + flows["S2E_9"]
            flows["I2R_9"] = prev["I"] - curr["I"] + flows["E2I_9"]

        For more complex flows (non-sequential, branching, looping,
        duplicating, etc.), users may override this method.

        :param dict state: A dictionary mapping compartment name to current
            tensor value. This should be updated in-place.
        :param t: A time-like index. During inference ``t`` may be either a
            slice (for vectorized inference) or an integer time index. During
            prediction ``t`` will be integer time index.
        :type t: int or slice
        :returns: A dict mapping flow name to tensor value.
        :rtype: dict
        """
        flows = {}
        flow = 0
        for source, destin in zip(self.compartments, self.compartments[1:] + ("R",)):
            flow = prev[source] - curr[source] + flow
            flows["{}2{}_{}".format(source, destin, t)] = flow
        return flows

    # Inference interface ########################################

    @torch.no_grad()
    @set_approx_sample_thresh(1000)
    def generate(self, fixed={}):
        """
        Generate data from the prior.

        :pram dict fixed: A dictionary of parameters on which to condition.
            These must be top-level parentless nodes, i.e. have no
            upstream stochastic dependencies.
        :returns: A dictionary mapping sample site name to sampled value.
        :rtype: dict
        """
        fixed = {k: torch.as_tensor(v) for k, v in fixed.items()}
        model = self._generative_model
        model = poutine.condition(model, fixed)
        trace = poutine.trace(model).get_trace()
        samples = OrderedDict((name, site["value"])
                              for name, site in trace.nodes.items()
                              if site["type"] == "sample")

        self._concat_series(samples, trace)
        return samples

    def fit_svi(self, *,
                num_samples=100,
                num_steps=2000,
                num_particles=32,
                learning_rate=0.1,
                learning_rate_decay=0.01,
                betas=(0.8, 0.99),
                haar=True,
                init_scale=0.01,
                guide_rank=0,
                jit=False,
                log_every=200,
                **options):
        """
        Runs stochastic variational inference to generate posterior samples.

        This runs :class:`~pyro.infer.svi.SVI`, setting the ``.samples``
        attribute on completion.

        This approximate inference method is useful for quickly iterating on
        probabilistic models.

        :param int num_samples: Number of posterior samples to draw from the
            trained guide. Defaults to 100.
        :param int num_steps: Number of :class:`~pyro.infer.svi.SVI` steps.
        :param int num_particles: Number of :class:`~pyro.infer.svi.SVI`
            particles per step.
        :param int learning_rate: Learning rate for the
            :class:`~pyro.optim.clipped_adam.ClippedAdam` optimizer.
        :param int learning_rate_decay: Learning rate for the
            :class:`~pyro.optim.clipped_adam.ClippedAdam` optimizer. Note this
            is decay over the entire schedule, not per-step decay.
        :param tuple betas: Momentum parameters for the
            :class:`~pyro.optim.clipped_adam.ClippedAdam` optimizer.
        :param bool haar: Whether to use a Haar wavelet reparameterizer.
        :param int guide_rank: Rank of the auto normal guide. If zero (default)
            use an :class:`~pyro.infer.autoguide.AutoNormal` guide. If a
            positive integer or None, use an
            :class:`~pyro.infer.autoguide.AutoLowRankMultivariateNormal` guide.
            If the string "full", use an
            :class:`~pyro.infer.autoguide.AutoMultivariateNormal` guide. These
            latter two require more ``num_steps`` to fit.
        :param float init_scale: Initial scale of the
            :class:`~pyro.infer.autoguide.AutoLowRankMultivariateNormal` guide.
        :param bool jit: Whether to use a jit compiled ELBO.
        :param int log_every: How often to log svi losses.
        :param int heuristic_num_particles: Passed to :meth:`heuristic` as
            ``num_particles``. Defaults to 1024.
        :returns: Time series of SVI losses (useful to diagnose convergence).
        :rtype: list
        """
        # Save configuration for .predict().
        self.relaxed = True
        self.num_quant_bins = 1

        # Setup Haar wavelet transform.
        if haar:
            time_dim = -2 if self.is_regional else -1
            dims = {"auxiliary": time_dim}
            supports = {"auxiliary": constraints.interval(-0.5, self.population + 0.5)}
            for name, (fn, is_regional) in self._non_compartmental.items():
                dims[name] = time_dim - fn.event_dim
                supports[name] = fn.support
            haar = _HaarSplitReparam(0, self.duration, dims, supports)

        # Heuristically initialize to feasible latents.
        heuristic_options = {k.replace("heuristic_", ""): options.pop(k)
                             for k in list(options)
                             if k.startswith("heuristic_")}
        assert not options, "unrecognized options: {}".format(", ".join(options))
        init_strategy = self._heuristic(haar, **heuristic_options)

        # Configure variational inference.
        logger.info("Running inference...")
        model = self._relaxed_model
        if haar:
            model = haar.reparam(model)
        if guide_rank == 0:
            guide = AutoNormal(model, init_loc_fn=init_strategy, init_scale=init_scale)
        elif guide_rank == "full":
            guide = AutoMultivariateNormal(model, init_loc_fn=init_strategy,
                                           init_scale=init_scale)
        elif guide_rank is None or isinstance(guide_rank, int):
            guide = AutoLowRankMultivariateNormal(model, init_loc_fn=init_strategy,
                                                  init_scale=init_scale, rank=guide_rank)
        else:
            raise ValueError("Invalid guide_rank: {}".format(guide_rank))
        Elbo = JitTrace_ELBO if jit else Trace_ELBO
        elbo = Elbo(max_plate_nesting=self.max_plate_nesting,
                    num_particles=num_particles, vectorize_particles=True,
                    ignore_jit_warnings=True)
        optim = ClippedAdam({"lr": learning_rate, "betas": betas,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        svi = SVI(model, guide, optim, elbo)

        # Run inference.
        start_time = default_timer()
        losses = []
        for step in range(1 + num_steps):
            loss = svi.step() / self.duration
            if step % log_every == 0:
                logger.info("step {} loss = {:0.4g}".format(step, loss))
            losses.append(loss)
        elapsed = default_timer() - start_time
        logger.info("SVI took {:0.1f} seconds, {:0.1f} step/sec"
                    .format(elapsed, (1 + num_steps) / elapsed))

        # Draw posterior samples.
        with torch.no_grad():
            particle_plate = pyro.plate("particles", num_samples,
                                        dim=-1 - self.max_plate_nesting)
            guide_trace = poutine.trace(particle_plate(guide)).get_trace()
            model_trace = poutine.trace(
                poutine.replay(particle_plate(model), guide_trace)).get_trace()
            self.samples = {name: site["value"] for name, site in model_trace.nodes.items()
                            if site["type"] == "sample"
                            if not site["is_observed"]
                            if not site_is_subsample(site)}
            if haar:
                haar.aux_to_user(self.samples)
        assert all(v.size(0) == num_samples for v in self.samples.values()), \
            {k: tuple(v.shape) for k, v in self.samples.items()}

        return losses

    @set_approx_log_prob_tol(0.1)
    def fit_mcmc(self, **options):
        r"""
        Runs NUTS inference to generate posterior samples.

        This uses the :class:`~pyro.infer.mcmc.nuts.NUTS` kernel to run
        :class:`~pyro.infer.mcmc.api.MCMC`, setting the ``.samples``
        attribute on completion.

        This uses an asymptotically exact enumeration-based model when
        ``num_quant_bins > 1``, and a cheaper moment-matched approximate model
        when ``num_quant_bins == 1``.

        :param \*\*options: Options passed to
            :class:`~pyro.infer.mcmc.api.MCMC`. The remaining options are
            pulled out and have special meaning.
        :param int num_samples: Number of posterior samples to draw via mcmc.
            Defaults to 100.
        :param int max_tree_depth: (Default 5). Max tree depth of the
            :class:`~pyro.infer.mcmc.nuts.NUTS` kernel.
        :param full_mass: Specification of mass matrix of the
            :class:`~pyro.infer.mcmc.nuts.NUTS` kernel. Defaults to full mass
            over global random variables.
        :param bool arrowhead_mass: Whether to treat ``full_mass`` as the head
            of an arrowhead matrix versus simply as a block. Defaults to False.
        :param int num_quant_bins: If greater than 1, use asymptotically exact
            inference via local enumeration over this many quantization bins.
            If equal to 1, use continuous-valued relaxed approximate inference.
            Note that computational cost is exponential in `num_quant_bins`.
            Defaults to 1 for relaxed inference.
        :param bool haar: Whether to use a Haar wavelet reparameterizer.
            Defaults to True.
        :param int haar_full_mass: Number of low frequency Haar components to
            include in the full mass matrix. If ``haar=False`` then this is
            ignored. Defaults to 10.
        :param int heuristic_num_particles: Passed to :meth:`heuristic` as
            ``num_particles``. Defaults to 1024.
        :returns: An MCMC object for diagnostics, e.g. ``MCMC.summary()``.
        :rtype: ~pyro.infer.mcmc.api.MCMC
        """
        _require_double_precision()

        # Parse options, saving some for use in .predict().
        num_samples = options.setdefault("num_samples", 100)
        num_chains = options.setdefault("num_chains", 1)
        self.num_quant_bins = options.pop("num_quant_bins", 1)
        assert isinstance(self.num_quant_bins, int)
        assert self.num_quant_bins >= 1
        self.relaxed = self.num_quant_bins == 1

        # Setup Haar wavelet transform.
        haar = options.pop("haar", False)
        haar_full_mass = options.pop("haar_full_mass", 10)
        full_mass = options.pop("full_mass", self.full_mass)
        assert isinstance(haar, bool)
        assert isinstance(haar_full_mass, int) and haar_full_mass >= 0
        assert isinstance(full_mass, (bool, list))
        haar_full_mass = min(haar_full_mass, self.duration)
        if not haar:
            haar_full_mass = 0
        if full_mass is True:
            haar_full_mass = 0  # No need to split.
        elif haar_full_mass >= self.duration:
            full_mass = True  # Effectively full mass.
            haar_full_mass = 0
        if haar:
            time_dim = -2 if self.is_regional else -1
            dims = {"auxiliary": time_dim}
            supports = {"auxiliary": constraints.interval(-0.5, self.population + 0.5)}
            for name, (fn, is_regional) in self._non_compartmental.items():
                dims[name] = time_dim - fn.event_dim
                supports[name] = fn.support
            haar = _HaarSplitReparam(haar_full_mass, self.duration, dims, supports)
        if haar_full_mass:
            assert full_mass and isinstance(full_mass, list)
            full_mass = full_mass[:]
            full_mass[0] += tuple(name + "_haar_split_0" for name in sorted(dims))

        # Heuristically initialize to feasible latents.
        heuristic_options = {k.replace("heuristic_", ""): options.pop(k)
                             for k in list(options)
                             if k.startswith("heuristic_")}
        init_strategy = init_to_generated(
            generate=functools.partial(self._heuristic, haar, **heuristic_options))

        # Configure a kernel.
        logger.info("Running inference...")
        model = self._relaxed_model if self.relaxed else self._quantized_model
        if haar:
            model = haar.reparam(model)
        kernel = NUTS(model,
                      full_mass=full_mass,
                      init_strategy=init_strategy,
                      max_plate_nesting=self.max_plate_nesting,
                      jit_compile=options.pop("jit_compile", False),
                      jit_options=options.pop("jit_options", None),
                      ignore_jit_warnings=options.pop("ignore_jit_warnings", True),
                      target_accept_prob=options.pop("target_accept_prob", 0.8),
                      max_tree_depth=options.pop("max_tree_depth", 5))
        if options.pop("arrowhead_mass", False):
            kernel.mass_matrix_adapter = ArrowheadMassMatrix()

        # Run mcmc.
        options.setdefault("disable_validation", None)
        mcmc = MCMC(kernel, **options)
        mcmc.run()
        self.samples = mcmc.get_samples()
        if haar:
            haar.aux_to_user(self.samples)

        # Unsqueeze samples to align particle dim for use in poutine.condition.
        # TODO refactor to an align_samples or particle_dim kwarg to MCMC.get_samples().
        model = self._relaxed_model if self.relaxed else self._quantized_model
        self.samples = align_samples(self.samples, model,
                                     particle_dim=-1 - self.max_plate_nesting)
        assert all(v.size(0) == num_samples * num_chains for v in self.samples.values()), \
            {k: tuple(v.shape) for k, v in self.samples.items()}

        return mcmc  # E.g. so user can run mcmc.summary().

    @torch.no_grad()
    @set_approx_log_prob_tol(0.1)
    @set_approx_sample_thresh(10000)
    def predict(self, forecast=0):
        """
        Predict latent variables and optionally forecast forward.

        This may be run only after :meth:`fit_mcmc` and draws the same
        ``num_samples`` as passed to :meth:`fit_mcmc`.

        :param int forecast: The number of time steps to forecast forward.
        :returns: A dictionary mapping sample site name (or compartment name)
            to a tensor whose first dimension corresponds to sample batching.
        :rtype: dict
        """
        if self.num_quant_bins > 1:
            _require_double_precision()
        if not self.samples:
            raise RuntimeError("Missing samples, try running .fit_mcmc() first")

        samples = self.samples
        num_samples = len(next(iter(samples.values())))
        particle_plate = pyro.plate("particles", num_samples,
                                    dim=-1 - self.max_plate_nesting)

        # Sample discrete auxiliary variables conditioned on the continuous
        # variables sampled by _quantized_model. This samples only time steps
        # [0:duration]. Here infer_discrete runs a forward-filter
        # backward-sample algorithm.
        logger.info("Predicting latent variables for {} time steps..."
                    .format(self.duration))
        model = self._sequential_model
        model = poutine.condition(model, samples)
        model = particle_plate(model)
        if not self.relaxed:
            model = infer_discrete(model, first_available_dim=-2 - self.max_plate_nesting)
        trace = poutine.trace(model).get_trace()
        samples = OrderedDict((name, site["value"].expand(site["fn"].shape()))
                              for name, site in trace.nodes.items()
                              if site["type"] == "sample"
                              if not site_is_subsample(site)
                              if not site_is_factor(site))
        assert all(v.size(0) == num_samples for v in samples.values()), \
            {k: tuple(v.shape) for k, v in samples.items()}

        # Optionally forecast with the forward _generative_model. This samples
        # time steps [duration:duration+forecast].
        if forecast:
            logger.info("Forecasting {} steps ahead...".format(forecast))
            model = self._generative_model
            model = poutine.condition(model, samples)
            model = particle_plate(model)
            trace = poutine.trace(model).get_trace(forecast)
            samples = OrderedDict((name, site["value"])
                                  for name, site in trace.nodes.items()
                                  if site["type"] == "sample"
                                  if not site_is_subsample(site)
                                  if not site_is_factor(site))

        self._concat_series(samples, trace, forecast)
        assert all(v.size(0) == num_samples for v in samples.values()), \
            {k: tuple(v.shape) for k, v in samples.items()}
        return samples

    @torch.no_grad()
    @set_approx_log_prob_tol(0.1)
    @set_approx_sample_thresh(100)  # This is robust to gross approximation.
    def heuristic(self, num_particles=1024, ess_threshold=0.5, retries=10):
        """
        Finds an initial feasible guess of all latent variables, consistent
        with observed data. This is needed because not all hypotheses are
        feasible and HMC needs to start at a feasible solution to progress.

        The default implementation attempts to find a feasible state using
        :class:`~pyro.infer.smcfilter.SMCFilter` with proprosals from the
        prior.  However this method may be overridden in cases where SMC
        performs poorly e.g. in high-dimensional models.

        :param int num_particles: Number of particles used for SMC.
        :param float ess_threshold: Effective sample size threshold for SMC.
        :returns: A dictionary mapping sample site name to tensor value.
        :rtype: dict
        """
        # Run SMC.
        model = _SMCModel(self)
        guide = _SMCGuide(self)
        for attempt in range(1, 1 + retries):
            smc = SMCFilter(model, guide, num_particles=num_particles,
                            ess_threshold=ess_threshold,
                            max_plate_nesting=self.max_plate_nesting)
            try:
                smc.init()
                for t in range(1, self.duration):
                    smc.step()
                break
            except SMCFailed as e:
                if attempt == retries:
                    raise
                logger.info("{}. Retrying...".format(e))
                continue

        # Select the most probable hypothesis.
        # Note this ignores the .finalize() likelihood.
        i = int(smc.state._log_weights.max(0).indices)
        init = {key: value[i, 0] for key, value in smc.state.items()}

        # Fill in sample site values.
        init = self.generate(init)
        aux = torch.stack([init[name] for name in self.compartments], dim=0)
        init["auxiliary"] = clamp(aux, min=0.5, max=self.population - 0.5)
        return init

    # Internal helpers ########################################

    def _heuristic(self, haar, **options):
        with poutine.block():
            init_values = self.heuristic(**options)
        assert isinstance(init_values, dict)
        assert "auxiliary" in init_values, \
            ".heuristic() did not define auxiliary value"
        if haar:
            haar.user_to_aux(init_values)
        logger.info("Heuristic init: {}".format(", ".join(
            "{}={:0.3g}".format(k, v.item())
            for k, v in sorted(init_values.items())
            if v.numel() == 1)))
        return init_to_value(values=init_values)

    def _concat_series(self, samples, trace, forecast=0):
        """
        Concatenate sequential time series into tensors, in-place.

        :param dict samples: A dictionary of samples.
        """
        time_dim = -2 if self.is_regional else -1
        for name in set(self.compartments).union(self.series):
            pattern = name + "_[0-9]+"
            series = []
            for key in list(samples):
                if re.match(pattern, key):
                    series.append(samples.pop(key))
            if not series:
                continue
            assert len(series) == self.duration + forecast
            series = torch.broadcast_tensors(*map(torch.as_tensor, series))
            dim = time_dim - trace.nodes[name + "_0"]["fn"].event_dim
            if series[0].dim() >= -dim:
                samples[name] = torch.cat(series, dim=dim)
            else:
                samples[name] = torch.stack(series)

    @lazy_property
    @torch.no_grad()
    def _non_compartmental(self):
        """
        A dict mapping name -> (distribution, is_regional) for all
        non-compartmental sites in :meth:`transition`. For simple models this
        is often empty; for time-heterogeneous models this may contain
        time-local latent variables.
        """
        # Trace a simple invocation of .transition().
        with torch.no_grad(), poutine.block():
            params = self.global_model()
            prev = self.initialize(params)
            for name in self.approximate:
                prev[name + "_approx"] = prev[name]
            curr = prev.copy()
            with poutine.trace() as tr:
                self.transition(params, curr, 0)
            flows = self.compute_flows(prev, curr, 0)

        # Extract latent variables that are not compartmental flows.
        result = OrderedDict()
        for name, site in tr.trace.iter_stochastic_nodes():
            if name in flows or site_is_subsample(site):
                continue
            assert name.endswith("_0"), name
            name = name[:-2]
            assert name in self.series, name
            # TODO This supports only the region_plate. For full plate support,
            # this could be replaced by a self.plate() method as in EasyGuide.
            is_regional = any(f.name == "region" for f in site["cond_indep_stack"])
            result[name] = site["fn"], is_regional
        return result

    def _sample_auxiliary(self):
        """
        Sample both compartmental and non-compartmental auxiliary variables.
        """
        C = len(self.compartments)
        T = self.duration
        R_shape = getattr(self.population, "shape", ())  # Region shape.

        # Sample the compartmental continuous reparameterizing variable.
        shape = (C, T) + R_shape
        auxiliary = pyro.sample("auxiliary",
                                dist.Uniform(-0.5, self.population + 0.5)
                                    .mask(False).expand(shape).to_event())
        extra_dims = auxiliary.dim() - len(shape)

        # Sample any non-compartmental time series in batch.
        non_compartmental = OrderedDict()
        for name, (fn, is_regional) in self._non_compartmental.items():
            fn = dist.ImproperUniform(fn.support, fn.batch_shape, fn.event_shape)
            shape = (T,)
            if self.is_regional:
                shape += R_shape if is_regional else (1,)
            # Manually expand, avoiding plates to enable HaarReparam and SplitReparam.
            non_compartmental[name] = pyro.sample(name, fn.expand(shape).to_event())

        # Move event dims to time_plate and region_plate dims.
        if extra_dims:  # If inside particle_plate.
            shape = auxiliary.shape[:1] + auxiliary.shape[extra_dims:]
            auxiliary = auxiliary.reshape(shape)
            for name, value in non_compartmental.items():
                shape = value.shape[:1] + value.shape[extra_dims:]
                non_compartmental[name] = value.reshape(shape)

        return auxiliary, non_compartmental

    def _transition_bwd(self, params, prev, curr, t):
        """
        Helper to collect probabilty factors from .transition() conditioned on
        previous and current enumerated states.
        """
        # Run .transition() conditioned on computed flows.
        cond_data = {"{}_{}".format(k, t): v for k, v in curr.items()}
        cond_data.update(self.compute_flows(prev, curr, t))
        with poutine.condition(data=cond_data):
            state = prev.copy()
            self.transition(params, state, t)  # Mutates state.

        # Validate that .transition() matches .compute_flows().
        if is_validation_enabled():
            for key in self.compartments:
                if not torch.allclose(state[key], curr[key]):
                    raise ValueError("Incorrect state['{}'] update in .transition(), "
                                     "check that .transition() matches .compute_flows()."
                                     .format(key))

    def _generative_model(self, forecast=0):
        """
        Forward generative model used for simulation and forecasting.
        """
        # Sample global parameters.
        params = self.global_model()

        # Sample initial values.
        state = self.initialize(params)
        state = {k: v if isinstance(v, torch.Tensor) else torch.tensor(float(v))
                 for k, v in state.items()}

        # Sequentially transition.
        for t in range(self.duration + forecast):
            for name in self.approximate:
                state[name + "_approx"] = state[name]
            self.transition(params, state, t)
            with self.region_plate:
                for name in self.compartments:
                    pyro.deterministic("{}_{}".format(name, t), state[name], event_dim=0)

        self._clear_plates()

    def _sequential_model(self):
        """
        Sequential model used to sample latents in the interval [0:duration].
        This is compatible with both quantized and relaxed inference.
        This method is called only inside particle_plate.
        This method is used only for prediction.
        """
        C = len(self.compartments)
        T = self.duration
        R_shape = getattr(self.population, "shape", ())  # Region shape.
        num_samples = len(next(iter(self.samples.values())))

        # Sample global parameters and auxiliary variables.
        params = self.global_model()
        auxiliary, non_compartmental = self._sample_auxiliary()

        # Reshape to accommodate the time_plate below.
        assert auxiliary.shape == (num_samples, C, T) + R_shape, \
            (auxiliary.shape, (num_samples, C, T) + R_shape)
        aux = [aux.unbind(2) for aux in auxiliary.unsqueeze(1).unbind(2)]

        # Sequentially transition.
        curr = self.initialize(params)
        for t in poutine.markov(range(T)):
            with self.region_plate:
                prev, curr = curr, {}

                # Extract any non-compartmental variables.
                for name, value in non_compartmental.items():
                    curr[name] = value[:, t:t+1]

                # Extract and enumerate all compartmental variables.
                for c, name in enumerate(self.compartments):
                    curr[name] = quantize("{}_{}".format(name, t), aux[c][t],
                                          min=0, max=self.population,
                                          num_quant_bins=self.num_quant_bins)
                    # Enable approximate inference by using aux as a
                    # non-enumerated proxy for enumerated compartment values.
                    if name in self.approximate:
                        curr[name + "_approx"] = aux[c][t]
                        prev.setdefault(name + "_approx", prev[name])

            self._transition_bwd(params, prev, curr, t)

        self._clear_plates()

    def _quantized_model(self):
        """
        Quantized vectorized model used for parallel-scan enumerated inference.
        This method is called only outside particle_plate.
        """
        C = len(self.compartments)
        T = self.duration
        Q = self.num_quant_bins
        R_shape = getattr(self.population, "shape", ())  # Region shape.

        # Sample global parameters and auxiliary variables.
        params = self.global_model()
        auxiliary, non_compartmental = self._sample_auxiliary()

        # Manually enumerate.
        curr, logp = quantize_enumerate(auxiliary, min=0, max=self.population,
                                        num_quant_bins=self.num_quant_bins)
        curr = OrderedDict(zip(self.compartments, curr.unbind(0)))
        logp = OrderedDict(zip(self.compartments, logp.unbind(0)))
        curr.update(non_compartmental)

        # Truncate final value from the right then pad initial value onto the left.
        init = self.initialize(params)
        prev = {}
        for name, value in init.items():
            if name in self.compartments:
                if isinstance(value, torch.Tensor):
                    value = value[..., None]  # Because curr is enumerated on the right.
                prev[name] = cat2(value, curr[name][:-1],
                                  dim=-3 if self.is_regional else -2)
            else:  # non-compartmental
                prev[name] = cat2(init[name], curr[name][:-1], dim=-curr[name].dim())

        # Reshape to support broadcasting, similar to EnumMessenger.
        def enum_reshape(tensor, position):
            assert tensor.size(-1) == Q
            assert tensor.dim() <= self.max_plate_nesting + 2
            tensor = tensor.permute(tensor.dim() - 1, *range(tensor.dim() - 1))
            shape = [Q] + [1] * (position + self.max_plate_nesting - (tensor.dim() - 2))
            shape.extend(tensor.shape[1:])
            return tensor.reshape(shape)

        for e, name in enumerate(self.compartments):
            curr[name] = enum_reshape(curr[name], e)
            logp[name] = enum_reshape(logp[name], e)
            prev[name] = enum_reshape(prev[name], e + C)

        # Enable approximate inference by using aux as a non-enumerated proxy
        # for enumerated compartment values.
        for name in self.approximate:
            aux = auxiliary[self.compartments.index(name)]
            curr[name + "_approx"] = aux
            prev[name + "_approx"] = cat2(init[name], aux[:-1],
                                          dim=-2 if self.is_regional else -1)

        # Record transition factors.
        with poutine.block(), poutine.trace() as tr:
            with self.time_plate:
                t = slice(0, T, 1)  # Used to slice data tensors.
                self._transition_bwd(params, prev, curr, t)
        tr.trace.compute_log_prob()
        for name, site in tr.trace.nodes.items():
            if site["type"] == "sample":
                log_prob = site["log_prob"]
                if log_prob.dim() <= self.max_plate_nesting:  # Not enumerated.
                    pyro.factor("transition_" + name, site["log_prob_sum"])
                    continue
                if self.is_regional and log_prob.shape[-1:] != R_shape:
                    # Poor man's tensor variable elimination.
                    log_prob = log_prob.expand(log_prob.shape[:-1] + R_shape) / R_shape[0]
                logp[name] = site["log_prob"]

        # Manually perform variable elimination.
        logp = reduce(operator.add, logp.values())
        logp = logp.reshape(Q ** C, Q ** C, T, -1)  # prev, curr, T, batch
        logp = logp.permute(3, 2, 0, 1).squeeze(0)  # batch, T, prev, curr
        logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)  # batch, prev, curr
        logp = logp.reshape(-1, Q ** C * Q ** C).logsumexp(-1).sum()
        warn_if_nan(logp)
        pyro.factor("transition", logp)

        # Apply final likelihood.
        prev = {name: prev[name + "_approx"] for name in self.approximate}
        curr = {name: curr[name + "_approx"] for name in self.approximate}
        with _disallow_latent_variables(".finalize()"):
            self.finalize(params, prev, curr)

        self._clear_plates()

    @set_relaxed_distributions()
    def _relaxed_model(self):
        """
        Relaxed vectorized model used for continuous inference.
        This method may be called either inside or outside particle_plate.
        """
        T = self.duration

        # Sample global parameters and auxiliary variables.
        params = self.global_model()
        auxiliary, non_compartmental = self._sample_auxiliary()
        particle_dims = auxiliary.dim() - (3 if self.is_regional else 2)
        assert particle_dims in (0, 1)

        # Split tensors into current state.
        curr = dict(zip(self.compartments, auxiliary.unbind(particle_dims)))
        curr.update(non_compartmental)

        # Truncate final value from the right then pad initial value onto the left.
        prev = {}
        for name, value in self.initialize(params).items():
            dim = particle_dims - curr[name].dim()
            t = (slice(None),) * particle_dims + (slice(0, -1),)
            prev[name] = cat2(value, curr[name][t], dim=dim)

        # Enable approximate inference.
        for name in self.approximate:
            curr[name + "_approx"] = curr[name]
            prev[name + "_approx"] = prev[name]

        # Transition.
        with self.time_plate:
            t = slice(0, T, 1)  # Used to slice data tensors.
            self._transition_bwd(params, prev, curr, t)

        # Apply final likelihood.
        with _disallow_latent_variables(".finalize()"):
            self.finalize(params, prev, curr)

        self._clear_plates()


class _SMCModel:
    """
    Helper to initialize a CompartmentalModel to a feasible initial state.
    """
    def __init__(self, model):
        assert isinstance(model, CompartmentalModel)
        self.model = model

    def init(self, state):
        with poutine.trace() as tr:
            params = self.model.global_model()
        for name, site in tr.trace.nodes.items():
            if site["type"] == "sample":
                state[name] = site["value"]

        self.t = 0
        state.update(self.model.initialize(params))
        self.step(state)  # Take one step since model.initialize is deterministic.

    def step(self, state):
        with poutine.block(), poutine.condition(data=state):
            params = self.model.global_model()

        with poutine.trace() as tr:
            # Temporarily extend state with approximations.
            extended_state = dict(state)
            for name in self.model.approximate:
                extended_state[name + "_approx"] = state[name]

            self.model.transition(params, extended_state, self.t)

            for name in self.model.approximate:
                del extended_state[name + "_approx"]
            state.update(extended_state)

        for name, site in tr.trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                state[name] = site["value"]
        self.t += 1


class _SMCGuide(_SMCModel):
    """
    Like _SMCModel but does not update state and does not observe.
    """
    def init(self, state):
        super().init(state.copy())

    def step(self, state):
        with poutine.block(hide_types=["observe"]):
            super().step(state.copy())


class _HaarSplitReparam:
    """
    Wrapper around ``HaarReparam`` and ``SplitReparam`` to additionally convert
    sample dicts between user-facing and auxiliary coordinates.
    """
    def __init__(self, split, duration, dims, supports):
        assert 0 <= split < duration
        self.split = split
        self.duration = duration
        self.dims = dims
        self.supports = supports

    def __bool__(self):
        return True

    def reparam(self, model):
        """
        Wrap a model with ``poutine.reparam``.
        """
        # Transform to Haar coordinates.
        config = {}
        for name, dim in self.dims.items():
            config[name] = HaarReparam(dim=dim, flip=True)
        model = poutine.reparam(model, config)

        if self.split:
            # Split into low- and high-frequency parts.
            splits = [self.split, self.duration - self.split]
            config = {}
            for name, dim in self.dims.items():
                config[name + "_haar"] = SplitReparam(splits, dim=dim)
            model = poutine.reparam(model, config)

        return model

    def user_to_aux(self, samples):
        """
        Convert from user-facing samples to auxiliary samples, in-place.
        """
        # Transform to Haar coordinates.
        for name, dim in self.dims.items():
            x = samples.pop(name)
            x = biject_to(self.supports[name]).inv(x)
            x = HaarTransform(dim=dim, flip=True)(x)
            samples[name + "_haar"] = x

        if self.split:
            # Split into low- and high-frequency parts.
            splits = [self.split, self.duration - self.split]
            for name, dim in self.dims.items():
                x0, x1 = samples.pop(name + "_haar").split(splits, dim=dim)
                samples[name + "_haar_split_0"] = x0
                samples[name + "_haar_split_1"] = x1

    def aux_to_user(self, samples):
        """
        Convert from auxiliary samples to user-facing samples, in-place.
        """
        if self.split:
            # Transform back from SplitReparam coordinates.
            for name, dim in self.dims.items():
                samples[name + "_haar"] = torch.cat([
                    samples.pop(name + "_haar_split_0"),
                    samples.pop(name + "_haar_split_1"),
                ], dim=dim)

        # Transform back from Haar coordinates.
        for name, dim in self.dims.items():
            x = samples.pop(name + "_haar")
            x = HaarTransform(dim=dim, flip=True).inv(x)
            x = biject_to(self.supports[name])(x)
            samples[name] = x
