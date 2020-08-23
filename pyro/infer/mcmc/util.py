# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import warnings
from collections import OrderedDict, defaultdict
from functools import partial, reduce
from itertools import product
import traceback as tb

import torch
from torch.distributions import biject_to
from opt_einsum import shared_intermediates

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape, logsumexp
from pyro.infer import config_enumerate
from pyro.infer.autoguide.initialization import InitMessenger, init_to_uniform
from pyro.infer.util import is_validation_enabled
from pyro.ops import stats
from pyro.ops.contract import contract_to_tensor
from pyro.ops.integrator import potential_grad
from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_site_shape, ignore_jit_warnings


class TraceTreeEvaluator:
    """
    Computes the log probability density of a trace (of a model with
    tree structure) that possibly contains discrete sample sites
    enumerated in parallel. This will be deprecated in favor of
    :class:`~pyro.infer.mcmc.util.EinsumTraceProbEvaluator`.

    :param model_trace: execution trace from a static model.
    :param bool has_enumerable_sites: whether the trace contains any
        discrete enumerable sites.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts.
    """
    def __init__(self,
                 model_trace,
                 has_enumerable_sites=False,
                 max_plate_nesting=None):
        self.has_enumerable_sites = has_enumerable_sites
        self.max_plate_nesting = max_plate_nesting
        # To be populated using the model trace once.
        self._log_probs = defaultdict(list)
        self._log_prob_shapes = defaultdict(tuple)
        self._children = defaultdict(list)
        self._enum_dims = {}
        self._plate_dims = {}
        self._parse_model_structure(model_trace)

    def _parse_model_structure(self, model_trace):
        if not self.has_enumerable_sites:
            return
        if self.max_plate_nesting is None:
            raise ValueError("Finite value required for `max_plate_nesting` when model "
                             "has discrete (enumerable) sites.")
        self._compute_log_prob_terms(model_trace)
        # 1. Infer model structure - compute parent-child relationship.
        sorted_ordinals = sorted(self._log_probs.keys())
        for i, child_node in enumerate(sorted_ordinals):
            for j in range(i-1, -1, -1):
                cur_node = sorted_ordinals[j]
                if cur_node < child_node:
                    self._children[cur_node].append(child_node)
                    break  # at most 1 parent.
        # 2. Populate `plate_dims` and `enum_dims` to be evaluated/
        #    enumerated out at each ordinal.
        self._populate_cache(frozenset(), frozenset(), set())

    def _populate_cache(self, ordinal, parent_ordinal, parent_enum_dims):
        """
        For each ordinal, populate the `plate` and `enum` dims to be
        evaluated or enumerated out.
        """
        log_prob_shape = self._log_prob_shapes[ordinal]
        plate_dims = sorted([frame.dim for frame in ordinal - parent_ordinal])
        enum_dims = set((i for i in range(-len(log_prob_shape), -self.max_plate_nesting)
                         if log_prob_shape[i] > 1))
        self._plate_dims[ordinal] = plate_dims
        self._enum_dims[ordinal] = set(enum_dims - parent_enum_dims)
        for c in self._children[ordinal]:
            self._populate_cache(c, ordinal, enum_dims)

    def _compute_log_prob_terms(self, model_trace):
        """
        Computes the conditional probabilities for each of the sites
        in the model trace, and stores the result in `self._log_probs`.
        """
        model_trace.compute_log_prob()
        self._log_probs = defaultdict(list)
        ordering = {name: frozenset(site["cond_indep_stack"])
                    for name, site in model_trace.nodes.items()
                    if site["type"] == "sample"}
        # Collect log prob terms per independence context.
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                self._log_probs[ordering[name]].append(site["log_prob"])
        if not self._log_prob_shapes:
            for ordinal, log_prob in self._log_probs.items():
                self._log_prob_shapes[ordinal] = broadcast_shape(*(t.shape for t in self._log_probs[ordinal]))

    def _reduce(self, ordinal, agg_log_prob=torch.tensor(0.)):
        """
        Reduce the log prob terms for the given ordinal:
          - taking log_sum_exp of factors in enum dims (i.e.
            adding up the probability terms).
          - summing up the dims within `max_plate_nesting`.
            (i.e. multiplying probs within independent batches).

        :param ordinal: node (ordinal)
        :param torch.Tensor agg_log_prob: aggregated `log_prob`
            terms from the downstream nodes.
        :return: `log_prob` with marginalized `plate` and `enum`
            dims.
        """
        log_prob = sum(self._log_probs[ordinal]) + agg_log_prob
        for enum_dim in self._enum_dims[ordinal]:
            log_prob = logsumexp(log_prob, dim=enum_dim, keepdim=True)
        for marginal_dim in self._plate_dims[ordinal]:
            log_prob = log_prob.sum(dim=marginal_dim, keepdim=True)
        return log_prob

    def _aggregate_log_probs(self, ordinal):
        """
        Aggregate the `log_prob` terms using depth first search.
        """
        if not self._children[ordinal]:
            return self._reduce(ordinal)
        agg_log_prob = sum(map(self._aggregate_log_probs, self._children[ordinal]))
        return self._reduce(ordinal, agg_log_prob)

    def log_prob(self, model_trace):
        """
        Returns the log pdf of `model_trace` by appropriately handling
        enumerated log prob factors.

        :return: log pdf of the trace.
        """
        with shared_intermediates():
            if not self.has_enumerable_sites:
                return model_trace.log_prob_sum()
            self._compute_log_prob_terms(model_trace)
            return self._aggregate_log_probs(ordinal=frozenset()).sum()


class TraceEinsumEvaluator:
    """
    Computes the log probability density of a trace (of a model with
    tree structure) that possibly contains discrete sample sites
    enumerated in parallel. This uses optimized `einsum` operations
    to marginalize out the the enumerated dimensions in the trace
    via :class:`~pyro.ops.contract.contract_to_tensor`.

    :param model_trace: execution trace from a static model.
    :param bool has_enumerable_sites: whether the trace contains any
        discrete enumerable sites.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts.
    """
    def __init__(self,
                 model_trace,
                 has_enumerable_sites=False,
                 max_plate_nesting=None):
        self.has_enumerable_sites = has_enumerable_sites
        self.max_plate_nesting = max_plate_nesting
        # To be populated using the model trace once.
        self._enum_dims = set()
        self.ordering = {}
        self._populate_cache(model_trace)

    def _populate_cache(self, model_trace):
        """
        Populate the ordinals (set of ``CondIndepStack`` frames)
        and enum_dims for each sample site.
        """
        if not self.has_enumerable_sites:
            return
        if self.max_plate_nesting is None:
            raise ValueError("Finite value required for `max_plate_nesting` when model "
                             "has discrete (enumerable) sites.")
        model_trace.compute_log_prob()
        model_trace.pack_tensors()
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and not isinstance(site["fn"], _Subsample):
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                self.ordering[name] = frozenset(model_trace.plate_to_symbol[f.name]
                                                for f in site["cond_indep_stack"]
                                                if f.vectorized)
        self._enum_dims = set(model_trace.symbol_to_dim) - set(model_trace.plate_to_symbol.values())

    def _get_log_factors(self, model_trace):
        """
        Aggregates the `log_prob` terms into a list for each
        ordinal.
        """
        model_trace.compute_log_prob()
        model_trace.pack_tensors()
        log_probs = OrderedDict()
        # Collect log prob terms per independence context.
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and not isinstance(site["fn"], _Subsample):
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                log_probs.setdefault(self.ordering[name], []).append(site["packed"]["log_prob"])
        return log_probs

    def log_prob(self, model_trace):
        """
        Returns the log pdf of `model_trace` by appropriately handling
        enumerated log prob factors.

        :return: log pdf of the trace.
        """
        if not self.has_enumerable_sites:
            return model_trace.log_prob_sum()
        log_probs = self._get_log_factors(model_trace)
        with shared_intermediates() as cache:
            return contract_to_tensor(log_probs, self._enum_dims, cache=cache)


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


class _PEMaker:
    def __init__(self, model, model_args, model_kwargs, trace_prob_evaluator, transforms):
        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.trace_prob_evaluator = trace_prob_evaluator
        self.transforms = transforms
        self._compiled_fn = None

    def _potential_fn(self, params):
        params_constrained = {k: self.transforms[k].inv(v) for k, v in params.items()}
        cond_model = poutine.condition(self.model, params_constrained)
        model_trace = poutine.trace(cond_model).get_trace(*self.model_args,
                                                          **self.model_kwargs)
        log_joint = self.trace_prob_evaluator.log_prob(model_trace)
        for name, t in self.transforms.items():
            log_joint = log_joint - torch.sum(
                t.log_abs_det_jacobian(params_constrained[name], params[name]))
        return -log_joint

    def _potential_fn_jit(self, skip_jit_warnings, jit_options, params):
        if not params:
            return self._potential_fn(params)
        names, vals = zip(*sorted(params.items()))

        if self._compiled_fn:
            return self._compiled_fn(*vals)

        with pyro.validation_enabled(False):
            tmp = []
            for _, v in pyro.get_param_store().named_parameters():
                if v.requires_grad:
                    v.requires_grad_(False)
                    tmp.append(v)

            def _pe_jit(*zi):
                params = dict(zip(names, zi))
                return self._potential_fn(params)

            if skip_jit_warnings:
                _pe_jit = ignore_jit_warnings()(_pe_jit)
            self._compiled_fn = torch.jit.trace(_pe_jit, vals, **jit_options)

            result = self._compiled_fn(*vals)
            for v in tmp:
                v.requires_grad_(True)
            return result

    def get_potential_fn(self, jit_compile=False, skip_jit_warnings=True, jit_options=None):
        if jit_compile:
            jit_options = {"check_trace": False} if jit_options is None else jit_options
            return partial(self._potential_fn_jit, skip_jit_warnings, jit_options)
        return self._potential_fn


def _find_valid_initial_params(model, model_args, model_kwargs, transforms, potential_fn,
                               prototype_params, max_tries_initial_params=100, num_chains=1,
                               init_strategy=init_to_uniform, trace=None):
    params = prototype_params

    # For empty models, exit early
    if not params:
        return params

    params_per_chain = defaultdict(list)
    num_found = 0
    model = InitMessenger(init_strategy)(model)
    for attempt in range(num_chains * max_tries_initial_params):
        if trace is None:
            trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
        samples = {name: trace.nodes[name]["value"].detach() for name in params}
        params = {k: transforms[k](v) for k, v in samples.items()}
        pe_grad, pe = potential_grad(potential_fn, params)

        if torch.isfinite(pe) and all(map(torch.all, map(torch.isfinite, pe_grad.values()))):
            for k, v in params.items():
                params_per_chain[k].append(v)
            num_found += 1
            if num_found == num_chains:
                if num_chains == 1:
                    return {k: v[0] for k, v in params_per_chain.items()}
                else:
                    return {k: torch.stack(v) for k, v in params_per_chain.items()}
        trace = None
    raise ValueError("Model specification seems incorrect - cannot find valid initial params.")


def initialize_model(model, model_args=(), model_kwargs={}, transforms=None, max_plate_nesting=None,
                     jit_compile=False, jit_options=None, skip_jit_warnings=False, num_chains=1,
                     init_strategy=init_to_uniform, initial_params=None):
    """
    Given a Python callable with Pyro primitives, generates the following model-specific
    properties needed for inference using HMC/NUTS kernels:

    - initial parameters to be sampled using a HMC kernel,
    - a potential function whose input is a dict of parameters in unconstrained space,
    - transforms to transform latent sites of `model` to unconstrained space,
    - a prototype trace to be used in MCMC to consume traces from sampled parameters.

    :param model: a Pyro model which contains Pyro primitives.
    :param tuple model_args: optional args taken by `model`.
    :param dict model_kwargs: optional kwargs taken by `model`.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is required if model contains
        discrete sample sites that can be enumerated over in parallel.
    :param bool jit_compile: Optional parameter denoting whether to use
        the PyTorch JIT to trace the log density computation, and use this
        optimized executable trace in the integrator.
    :param dict jit_options: A dictionary contains optional arguments for
        :func:`torch.jit.trace` function.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.
    :param int num_chains: Number of parallel chains. If `num_chains > 1`,
        the returned `initial_params` will be a list with `num_chains` elements.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param dict initial_params: dict containing initial tensors in unconstrained
        space to initiate the markov chain.
    :returns: a tuple of (`initial_params`, `potential_fn`, `transforms`, `prototype_trace`)
    """
    # XXX `transforms` domains are sites' supports
    # FIXME: find a good pattern to deal with `transforms` arg
    if transforms is None:
        automatic_transform_enabled = True
        transforms = {}
    else:
        automatic_transform_enabled = False
    if max_plate_nesting is None:
        max_plate_nesting = _guess_max_plate_nesting(model, model_args, model_kwargs)
    # Wrap model in `poutine.enum` to enumerate over discrete latent sites.
    # No-op if model does not have any discrete latents.
    model = poutine.enum(config_enumerate(model),
                         first_available_dim=-1 - max_plate_nesting)
    prototype_model = poutine.trace(InitMessenger(init_strategy)(model))
    model_trace = prototype_model.get_trace(*model_args, **model_kwargs)
    has_enumerable_sites = False
    prototype_samples = {}
    for name, node in model_trace.iter_stochastic_nodes():
        fn = node["fn"]
        if isinstance(fn, _Subsample):
            if fn.subsample_size is not None and fn.subsample_size < fn.size:
                raise NotImplementedError("HMC/NUTS does not support model with subsample sites.")
            continue
        if node["fn"].has_enumerate_support:
            has_enumerable_sites = True
            continue
        # we need to detach here because this sample can be a leaf variable,
        # so we can't change its requires_grad flag to calculate its grad in
        # velocity_verlet
        prototype_samples[name] = node["value"].detach()
        if automatic_transform_enabled:
            transforms[name] = biject_to(node["fn"].support).inv

    trace_prob_evaluator = TraceEinsumEvaluator(model_trace,
                                                has_enumerable_sites,
                                                max_plate_nesting)

    pe_maker = _PEMaker(model, model_args, model_kwargs, trace_prob_evaluator, transforms)

    if initial_params is None:
        prototype_params = {k: transforms[k](v) for k, v in prototype_samples.items()}
        # Note that we deliberately do not exercise jit compilation here so as to
        # enable potential_fn to be picklable (a torch._C.Function cannot be pickled).
        # We pass model_trace merely for computational savings.
        initial_params = _find_valid_initial_params(model, model_args, model_kwargs, transforms,
                                                    pe_maker.get_potential_fn(), prototype_params,
                                                    num_chains=num_chains, init_strategy=init_strategy,
                                                    trace=model_trace)
    potential_fn = pe_maker.get_potential_fn(jit_compile, skip_jit_warnings, jit_options)
    return initial_params, potential_fn, transforms, model_trace


def _safe(fn):
    """
    Safe version of utilities in the :mod:`pyro.ops.stats` module. Wrapped
    functions return `NaN` tensors instead of throwing exceptions.

    :param fn: stats function from :mod:`pyro.ops.stats` module.
    """
    @functools.wraps(fn)
    def wrapped(sample, *args, **kwargs):
        try:
            val = fn(sample, *args, **kwargs)
        except Exception:
            warnings.warn(tb.format_exc())
            val = torch.full(sample.shape[2:], float("nan"),
                             dtype=sample.dtype, device=sample.device)
        return val

    return wrapped


def diagnostics(samples, group_by_chain=True):
    """
    Gets diagnostics statistics such as effective sample size and
    split Gelman-Rubin using the samples drawn from the posterior
    distribution.

    :param dict samples: dictionary of samples keyed by site name.
    :param bool group_by_chain: If True, each variable in `samples`
        will be treated as having shape `num_chains x num_samples x sample_shape`.
        Otherwise, the corresponding shape will be `num_samples x sample_shape`
        (i.e. without chain dimension).
    :return: dictionary of diagnostic stats for each sample site.
    """
    diagnostics = {}
    for site, support in samples.items():
        if not group_by_chain:
            support = support.unsqueeze(0)
        site_stats = OrderedDict()
        site_stats["n_eff"] = _safe(stats.effective_sample_size)(support)
        site_stats["r_hat"] = stats.split_gelman_rubin(support)
        diagnostics[site] = site_stats
    return diagnostics


def summary(samples, prob=0.9, group_by_chain=True):
    """
    Returns a summary table displaying diagnostics of ``samples`` from the
    posterior. The diagnostics displayed are mean, standard deviation, median,
    the 90% Credibility Interval, :func:`~pyro.ops.stats.effective_sample_size`,
    :func:`~pyro.ops.stats.split_gelman_rubin`.

    :param dict samples: dictionary of samples keyed by site name.
    :param float prob: the probability mass of samples within the credibility interval.
    :param bool group_by_chain: If True, each variable in `samples`
        will be treated as having shape `num_chains x num_samples x sample_shape`.
        Otherwise, the corresponding shape will be `num_samples x sample_shape`
        (i.e. without chain dimension).
    """
    if not group_by_chain:
        samples = {k: v.unsqueeze(0) for k, v in samples.items()}

    summary_dict = {}
    for name, value in samples.items():
        value_flat = torch.reshape(value, (-1,) + value.shape[2:])
        mean = value_flat.mean(dim=0)
        std = value_flat.std(dim=0)
        median = value_flat.median(dim=0)[0]
        hpdi = stats.hpdi(value_flat, prob=prob)
        n_eff = _safe(stats.effective_sample_size)(value)
        r_hat = stats.split_gelman_rubin(value)
        hpd_lower = '{:.1f}%'.format(50 * (1 - prob))
        hpd_upper = '{:.1f}%'.format(50 * (1 + prob))
        summary_dict[name] = OrderedDict([("mean", mean), ("std", std), ("median", median),
                                          (hpd_lower, hpdi[0]), (hpd_upper, hpdi[1]),
                                          ("n_eff", n_eff), ("r_hat", r_hat)])
    return summary_dict


def print_summary(samples, prob=0.9, group_by_chain=True):
    """
    Prints a summary table displaying diagnostics of ``samples`` from the
    posterior. The diagnostics displayed are mean, standard deviation, median,
    the 90% Credibility Interval, :func:`~pyro.ops.stats.effective_sample_size`,
    :func:`~pyro.ops.stats.split_gelman_rubin`.

    :param dict samples: dictionary of samples keyed by site name.
    :param float prob: the probability mass of samples within the credibility interval.
    :param bool group_by_chain: If True, each variable in `samples`
        will be treated as having shape `num_chains x num_samples x sample_shape`.
        Otherwise, the corresponding shape will be `num_samples x sample_shape`
        (i.e. without chain dimension).
    """
    if len(samples) == 0:
        return
    summary_dict = summary(samples, prob, group_by_chain)

    row_names = {k: k + '[' + ','.join(map(lambda x: str(x - 1), v.shape[2:])) + ']'
                 for k, v in samples.items()}
    max_len = max(max(map(lambda x: len(x), row_names.values())), 10)
    name_format = '{:>' + str(max_len) + '}'
    header_format = name_format + ' {:>9}' * 7
    columns = [''] + list(list(summary_dict.values())[0].keys())

    print()
    print(header_format.format(*columns))

    row_format = name_format + ' {:>9.2f}' * 7
    for name, stats_dict in summary_dict.items():
        shape = stats_dict["mean"].shape
        if len(shape) == 0:
            print(row_format.format(name, *stats_dict.values()))
        else:
            for idx in product(*map(range, shape)):
                idx_str = '[{}]'.format(','.join(map(str, idx)))
                print(row_format.format(name + idx_str, *[v[idx] for v in stats_dict.values()]))
    print()


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


def predictive(model, posterior_samples, *args, **kwargs):
    """
    .. warning::
        This function is deprecated and will be removed in a future release.
        Use the :class:`~pyro.infer.predictive.Predictive` class instead.

    Run model by sampling latent parameters from `posterior_samples`, and return
    values at sample sites from the forward run. By default, only sites not contained in
    `posterior_samples` are returned. This can be modified by changing the `return_sites`
    keyword argument.

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
        * **return_trace** (``bool``) - whether to return the full trace. Note that this is vectorized
          over `num_samples`.
        * **parallel** (``bool``) - predict in parallel by wrapping the existing model
          in an outermost `plate` messenger. Note that this requires that the model has
          all batch dims correctly annotated via :class:`~pyro.plate`. Default is `False`.

    :return: dict of samples from the predictive distribution, or a single vectorized
        `trace` (if `return_trace=True`).
    """
    warnings.warn('The `mcmc.predictive` function is deprecated and will be removed in '
                  'a future release. Use the `pyro.infer.Predictive` class instead.',
                  FutureWarning)
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
