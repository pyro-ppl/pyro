# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from operator import attrgetter
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Union

import torch
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.distributions.util import eye_like, is_identically_zero
from pyro.infer.inspect import get_dependencies
from pyro.nn.module import PyroModule, PyroParam

from .guides import AutoGuide
from .initialization import InitMessenger, init_to_feasible
from .utils import deep_setattr, helpful_support_errors


def _config_auxiliary(msg):
    return {"is_auxiliary": True}


class AutoStructured(AutoGuide):
    """
    Structured guide whose conditional distributions are Delta, Normal,
    MultivariateNormal, or by a callable, and whose latent variables can depend
    on each other either linearly (in unconstrained space) or via shearing by a
    callable.

    Usage::

        def model(data):
            x = pyro.sample("x", dist.LogNormal(0, 1))
            with pyro.plate("plate", len(data)):
                y = pyro.sample("y", dist.Normal(0, 1))
                pyro.sample("z", dist.Normal(y, x), obs=data)

        # Either fully automatic...
        guide = AutoStructured(model)

        # ...or with specified conditional and dependency types...
        guide = AutoStructured(
            model, conditionals="normal", dependencies="linear"
        )

        # ...or with custom dependency structure and distribution types.
        guide = AutoStructured(
            model=model,
            conditionals={"x": "normal", "y": "delta"},
            dependencies={"x": {"y": "linear"}},
        )

    Once trained, this guide can be used with
    :class:`~pyro.infer.reparam.structured.StructuredReparam` to precondition a
    model for use in HMC and NUTS inference.

    .. note:: If you declare a dependency of a high-dimensional downstream
        variable on a low-dimensional upstream variable, you may want to use
        a lower learning rate for that weight, e.g.::

            def optim_config(param_name):
                config = {"lr": 0.01}
                if "deps.my_downstream.my_upstream" in param_name:
                    config["lr"] *= 0.1
                return config

            adam = pyro.optim.Adam(optim_config)

    :param callable model: A Pyro model.
    :param conditionals: Either a single distribution type or a dict mapping
        each latent variable name to a distribution type. A distribution type
        is either a string in {"delta", "normal", "mvn"} or a callable that
        returns a sample from a zero mean (or approximately centered) noise
        distribution (such callables typically call ``pyro.param()`` and
        ``pyro.sample()`` internally).
    :param dependencies: Dependency type, or a dict mapping each site name to a
        dict mapping its upstream dependencies to dependency types. If only a
        dependecy type is provided, dependency structure will be inferred. A
        dependency type is either the string "linear" or a callable that maps a
        *flattened* upstream perturbation to *flattened* downstream
        perturbation. The string "linear" is equivalent to
        ``nn.Linear(upstream.numel(), downstream.numel(), bias=False)``.
        Dependencies must not contain cycles or self-loops.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    scale_constraint = constraints.softplus_positive
    scale_tril_constraint = constraints.softplus_lower_cholesky

    def __init__(
        self,
        model,
        *,
        conditionals: Union[str, Dict[str, Union[str, Callable]]] = "mvn",
        dependencies: Union[str, Dict[str, Dict[str, Union[str, Callable]]]] = "linear",
        init_loc_fn: Callable = init_to_feasible,
        init_scale: float = 0.1,
        create_plates: Optional[Callable] = None,
    ):
        assert isinstance(conditionals, (dict, str))
        if isinstance(conditionals, dict):
            for name, fn in conditionals.items():
                assert isinstance(name, str)
                assert isinstance(fn, str) or callable(fn)
        assert isinstance(dependencies, (dict, str))
        if isinstance(dependencies, dict):
            for downstream, deps in dependencies.items():
                assert downstream in conditionals
                assert isinstance(deps, dict)
                for upstream, dep in deps.items():
                    assert upstream in conditionals
                    assert upstream != downstream
                    assert isinstance(dep, str) or callable(dep)
        self.conditionals = conditionals
        self.dependencies = dependencies

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(f"Expected init_scale > 0. but got {init_scale}")
        self._init_scale = init_scale
        self._original_model = (model,)
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _auto_config(self, sample_sites, args, kwargs):
        # Instantiate conditionals as dictionaries.
        if not isinstance(self.conditionals, dict):
            self.conditionals = {
                name: self.conditionals for name, site in sample_sites.items()
            }

        # Instantiate dependencies as dictionaries.
        if not isinstance(self.dependencies, dict):
            model = self._original_model[0]
            meta = poutine.block(get_dependencies)(model, args, kwargs)
            # Use posterior dependency edges but with prior ordering. This
            # allows sampling of globals before locals on which they depend.
            prior_order = {name: i for i, name in enumerate(sample_sites)}
            dependencies = defaultdict(dict)
            for d, upstreams in meta["posterior_dependencies"].items():
                assert d in sample_sites
                for u, plates in upstreams.items():
                    # TODO use plates to reduce dimension of dependency.
                    if u in sample_sites:
                        if prior_order[u] > prior_order[d]:
                            dependencies[u][d] = self.dependencies
                        elif prior_order[d] > prior_order[u]:
                            dependencies[d][u] = self.dependencies
            self.dependencies = dict(dependencies)
        self._original_model = None

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self.locs = PyroModule()
        self.scales = PyroModule()
        self.scale_trils = PyroModule()
        self.conds = PyroModule()
        self.deps = PyroModule()
        self._batch_shapes = {}
        self._unconstrained_event_shapes = {}
        sample_sites = OrderedDict(self.prototype_trace.iter_stochastic_nodes())
        self._auto_config(sample_sites, args, kwargs)

        # Collect unconstrained shapes.
        init_locs = {}
        numel = {}
        for name, site in sample_sites.items():
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support).inv(site["value"].detach()).detach()
                )
            self._batch_shapes[name] = site["fn"].batch_shape
            self._unconstrained_event_shapes[name] = init_loc.shape[
                len(site["fn"].batch_shape) :
            ]
            numel[name] = init_loc.numel()
            init_locs[name] = init_loc.reshape(-1)

        # Initialize guide params.
        children = defaultdict(list)
        num_pending = {}
        for name, site in sample_sites.items():
            # Initialize location parameters.
            init_loc = init_locs[name]
            deep_setattr(self.locs, name, PyroParam(init_loc))

            # Initialize parameters of conditional distributions.
            conditional = self.conditionals[name]
            if callable(conditional):
                deep_setattr(self.conds, name, conditional)
            else:
                if conditional not in ("delta", "normal", "mvn"):
                    raise ValueError(f"Unsupported conditional type: {conditional}")
                if conditional in ("normal", "mvn"):
                    init_scale = torch.full_like(init_loc, self._init_scale)
                    deep_setattr(
                        self.scales, name, PyroParam(init_scale, self.scale_constraint)
                    )
                if conditional == "mvn":
                    init_scale_tril = eye_like(init_loc, init_loc.numel())
                    deep_setattr(
                        self.scale_trils,
                        name,
                        PyroParam(init_scale_tril, self.scale_tril_constraint),
                    )

            # Initialize dependencies on upstream variables.
            num_pending[name] = 0
            deps = PyroModule()
            deep_setattr(self.deps, name, deps)
            for upstream, dep in self.dependencies.get(name, {}).items():
                assert upstream in sample_sites
                children[upstream].append(name)
                num_pending[name] += 1
                if isinstance(dep, str) and dep == "linear":
                    dep = torch.nn.Linear(numel[upstream], numel[name], bias=False)
                    dep.weight.data.zero_()
                elif not callable(dep):
                    raise ValueError(
                        f"Expected either the string 'linear' or a callable, but got {dep}"
                    )
                deep_setattr(deps, upstream, dep)

        # Topologically sort sites.
        # TODO should we choose a more optimal structure?
        self._sorted_sites = []
        while num_pending:
            name, count = min(num_pending.items(), key=lambda kv: (kv[1], kv[0]))
            assert count == 0, f"cyclic dependency: {name}"
            del num_pending[name]
            for child in children[name]:
                num_pending[child] -= 1
            site = self._compress_site(sample_sites[name])
            self._sorted_sites.append((name, site))

        # Prune non-essential parts of the trace to save memory.
        for name, site in self.prototype_trace.nodes.items():
            site.clear()

    @staticmethod
    def _compress_site(site):
        # Save memory by retaining only necessary parts of the site.
        return {
            "name": site["name"],
            "type": site["type"],
            "cond_indep_stack": site["cond_indep_stack"],
            "fn": SimpleNamespace(
                support=site["fn"].support,
                event_dim=site["fn"].event_dim,
            ),
        }

    @poutine.infer_config(config_fn=_config_auxiliary)
    def get_deltas(self, save_params=None):
        deltas = {}
        aux_values = {}
        compute_density = poutine.get_mask() is not False
        for name, site in self._sorted_sites:
            if save_params is not None and name not in save_params:
                continue

            # Sample zero-mean blockwise independent Delta/Normal/MVN.
            log_density = 0.0
            loc = attrgetter(name)(self.locs)
            zero = torch.zeros_like(loc)
            conditional = self.conditionals[name]
            if callable(conditional):
                aux_value = attrgetter(name)(self.conds)()
            elif conditional == "delta":
                aux_value = zero
            elif conditional == "normal":
                aux_value = pyro.sample(
                    name + "_aux",
                    dist.Normal(zero, 1).to_event(1),
                    infer={"is_auxiliary": True},
                )
                scale = attrgetter(name)(self.scales)
                aux_value = aux_value * scale
                if compute_density:
                    log_density = (-scale.log()).expand_as(aux_value)
            elif conditional == "mvn":
                # This overparametrizes by learning (scale,scale_tril),
                # enabling faster learning of the more-global scale parameter.
                aux_value = pyro.sample(
                    name + "_aux",
                    dist.Normal(zero, 1).to_event(1),
                    infer={"is_auxiliary": True},
                )
                scale = attrgetter(name)(self.scales)
                scale_tril = attrgetter(name)(self.scale_trils)
                aux_value = aux_value @ scale_tril.T * scale
                if compute_density:
                    log_density = (
                        -scale_tril.diagonal(dim1=-2, dim2=-1).log() - scale.log()
                    ).expand_as(aux_value)
            else:
                raise ValueError(f"Unsupported conditional type: {conditional}")

            # Accumulate upstream dependencies.
            # Note: by accumulating upstream dependencies before updating the
            # aux_values dict, we encode a block-sparse structure of the
            # precision matrix; if we had instead accumulated after updating
            # aux_values, we would encode a block-sparse structure of the
            # covariance matrix.
            # Note: these shear transforms have no effect on the Jacobian
            # determinant, and can therefore be excluded from the log_density
            # computation below, even for nonlinear dep().
            deps = attrgetter(name)(self.deps)
            for upstream in self.dependencies.get(name, {}):
                dep = attrgetter(upstream)(deps)
                aux_value = aux_value + dep(aux_values[upstream])
            aux_values[name] = aux_value

            # Shift by loc and reshape.
            batch_shape = torch.broadcast_shapes(
                aux_value.shape[:-1], self._batch_shapes[name]
            )
            unconstrained = (aux_value + loc).reshape(
                batch_shape + self._unconstrained_event_shapes[name]
            )
            if not is_identically_zero(log_density):
                log_density = log_density.reshape(batch_shape + (-1,)).sum(-1)

            # Transform to constrained space.
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained)
            if compute_density and conditional != "delta":
                assert transform.codomain.event_dim == site["fn"].event_dim
                log_density = log_density + transform.inv.log_abs_det_jacobian(
                    value, unconstrained
                )

            # Create a reparametrized Delta distribution.
            deltas[name] = dist.Delta(value, log_density, site["fn"].event_dim)

        return deltas

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        deltas = self.get_deltas()
        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self._sorted_sites:
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                result[name] = pyro.sample(name, deltas[name])

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        result = {}
        for name, site in self._sorted_sites:
            loc = attrgetter(name)(self.locs).detach()
            shape = self._batch_shapes[name] + self._unconstrained_event_shapes[name]
            loc = loc.reshape(shape)
            result[name] = biject_to(site["fn"].support)(loc)
        return result
