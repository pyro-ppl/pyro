# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from typing import Callable, Dict, Optional, Set, Tuple, Union

import torch
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.infer.inspect import get_dependencies
from pyro.nn.module import PyroModule, PyroParam
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.runtime import am_i_wrapped, get_plates
from pyro.poutine.util import site_is_subsample

from .guides import AutoGuide
from .initialization import InitMessenger, init_to_feasible
from .utils import deep_getattr, deep_setattr, helpful_support_errors


class AutoGaussian(AutoGuide):
    """
    Gaussian guide with optimal conditional independence structure.

    This is equivalent to a full rank :class:`AutoMultivariateNormal` guide,
    but with a sparse precision matrix determined by dependencies and plates in
    the model [1]. Depending on model structure, this can have asymptotically
    better statistical efficiency than :class:`AutoMultivariateNormal` .

    The default "dense" backend should have similar computational complexity to
    :class:`AutoMultivariateNormal` . The experimental "funsor" backend can be
    asymptotically cheaper in terms of time and space (using Gaussian tensor
    variable elimination [2,3]), but incurs large constant overhead. The
    "funsor" backend requires `funsor <https://funsor.pyro.ai>`_ which can be
    installed via ``pip install pyro-ppl[funsor]``.

    The guide currently does not depend on the model's ``*args, **kwargs``.

    Example::

        guide = AutoGaussian(model)
        svi = SVI(model, guide, ...)

    Example using funsor backend::

        !pip install pyro-ppl[funsor]
        guide = AutoGaussian(model, backend="funsor")
        svi = SVI(model, guide, ...)

    **References**

    [1] S.Webb, A.Goliński, R.Zinkov, N.Siddharth, T.Rainforth, Y.W.Teh, F.Wood (2018)
        "Faithful inversion of generative models for effective amortized inference"
        https://dl.acm.org/doi/10.5555/3327144.3327229
    [2] F.Obermeyer, E.Bingham, M.Jankowiak, J.Chiu, N.Pradhan, A.M.Rush, N.Goodman
        (2019)
        "Tensor Variable Elimination for Plated Factor Graphs"
        http://proceedings.mlr.press/v97/obermeyer19a/obermeyer19a.pdf
    [3] F. Obermeyer, E. Bingham, M. Jankowiak, D. Phan, J. P. Chen
        (2019)
        "Functional Tensors for Probabilistic Programming"
        https://arxiv.org/abs/1910.10775

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    :param str backend: Back end for performing Gaussian tensor variable
        elimination. Defaults to "dense"; other options include "funsor".
    """

    # Class configurable parameters.
    default_backend: str = "dense"
    scale_constraint = constraints.softplus_positive

    # Type hints for instance variables.
    backend: str
    locs: PyroModule
    scales: PyroModule
    precision_chols: PyroModule
    _sorted_sites: Dict[str, Dict[str, object]]
    _init_scale: float
    _original_model: Tuple[Callable]
    _unconstrained_event_shapes: Dict[str, torch.Size]
    _broken_event_shapes: Dict[str, torch.Size]
    _broken_plates: Dict[str, Tuple[str, ...]]

    def __init__(
        self,
        model: Callable,
        *,
        init_loc_fn: Callable = init_to_feasible,
        init_scale: float = 0.1,
        create_plates: Optional[Callable] = None,
        backend=None,
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(f"Expected init_scale > 0. but got {init_scale}")
        self._init_scale = init_scale
        self._original_model = (model,)
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)
        self.backend = self.default_backend if backend is None else backend

    def _setup_prototype(self, *args, **kwargs) -> None:
        super()._setup_prototype(*args, **kwargs)

        self.locs = PyroModule()
        self.scales = PyroModule()
        self.precision_chols = PyroModule()
        self._sorted_sites = OrderedDict()
        self._unconstrained_event_shapes = {}

        model = self._original_model[0]
        meta = poutine.block(get_dependencies)(model, args, kwargs)
        self.dependencies = meta["prior_dependencies"]
        for name, site in self.prototype_trace.nodes.items():
            if site["type"] == "sample":
                if not site_is_subsample(site):
                    self._sorted_sites[name] = site
        for d, site in self._sorted_sites.items():
            precision_size = 0
            precision_plates: Set[CondIndepStackFrame] = set()
            if not site["is_observed"]:
                # Initialize latent variable location-scale parameters.
                # The scale parameters are statistically redundant, but improve
                # learning with coordinate-wise optimizers.
                with helpful_support_errors(site):
                    init_loc = biject_to(site["fn"].support).inv(site["value"]).detach()
                init_scale = torch.full_like(init_loc, self._init_scale)
                batch_shape = site["fn"].batch_shape
                event_shape = init_loc.shape[len(batch_shape) :]
                self._unconstrained_event_shapes[d] = event_shape
                event_dim = len(event_shape)
                deep_setattr(self.locs, d, PyroParam(init_loc, event_dim=event_dim))
                deep_setattr(
                    self.scales,
                    d,
                    PyroParam(
                        init_scale,
                        constraint=self.scale_constraint,
                        event_dim=event_dim,
                    ),
                )

                # Gather shapes for precision matrices.
                for f in site["cond_indep_stack"]:
                    if f.vectorized:
                        precision_plates.add(f)
                precision_size.add(event_shape.numel())

            # Initialize precision matrices.
            # This adds a batched dense matrix for each factor, achieving
            # statistically optimal sparsity structure of the model's joint
            # precision matrix. Multiple factors may redundantly parametrize
            # entries of the precision matrix on which they overlap, incurring
            # slight computational cost but no cost to statistical efficiency.
            for u in self.dependencies[d]:
                u_site = self.prototype_trace.nodes[u]
                u_numel = self._unconstrained_event_shapes[u].numel()
                for f in u_site["cond_indep_stack"]:
                    if f.vectorized:
                        if f in site["cond_indep_stack"]:
                            precision_plates.add(f)
                        else:
                            u_numel *= f.size
                precision_size += u_numel
            batch_shape = torch.Size(
                f.size for f in sorted(precision_plates, key=lambda f: f.dim)
            )
            eye = torch.eye(precision_size) + torch.zeros(batch_shape + (1, 1))
            deep_setattr(
                self.precision_chols,
                d,
                PyroParam(eye, constraint=constraints.lower_cholesky, event_dim=2),
            )

        # Dispatch to backend logic.
        backend_fn = getattr(self, f"_{self.backend}_setup_prototype", None)
        if backend_fn is not None:
            backend_fn(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        aux_values, global_log_density = self._sample_aux_values()
        values, log_densities = self._transform_values(aux_values)

        # Replay via Pyro primitives.
        plates = self._create_plates(*args, **kwargs)
        for name, site in self._sorted_sites.items():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                pyro.sample(
                    name,
                    dist.Delta(values[name], log_densities[name], site["fn"].event_dim),
                )
        if am_i_wrapped() and poutine.get_mask() is not False:
            pyro.factor(self._pyro_name, global_log_density)
        return values

    def median(self) -> Dict[str, torch.Tensor]:
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        with torch.no_grad(), poutine.mask(mask=False):
            aux_values = {name: 0.0 for name in self._sorted_sites}
            values, _ = self._transform_values(aux_values)
        return values

    def _transform_values(
        self,
        aux_values: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Union[float, torch.Tensor]]:
        # Learnably transform auxiliary values to user-facing values.
        values = {}
        log_densities = defaultdict(float)
        compute_density = am_i_wrapped() and poutine.get_mask() is not False
        for name, site in self._sorted_sites.items():
            loc = deep_getattr(self.locs, name)
            scale = deep_getattr(self.scales, name)
            unconstrained = aux_values[name] * scale + loc

            # Transform to constrained space.
            transform = biject_to(site["fn"].support)
            values[name] = transform(unconstrained)
            if compute_density:
                assert transform.codomain.event_dim == site["fn"].event_dim
                log_densities[name] = transform.inv.log_abs_det_jacobian(
                    values[name], unconstrained
                ) - scale.log().reshape(site["fn"].batch_shape + (-1,)).sum(-1)

        return values, log_densities

    def _sample_aux_values(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Union[float, torch.Tensor]]:
        # Dispatch to backend logic.
        backend_fn = getattr(self, f"_{self.backend}_sample_aux_values", None)
        if backend_fn is None:
            raise NotImplementedError(f"Unknown AutoGaussian backend: {self.backend}")
        return backend_fn()

    ############################################################################
    # Dense backend

    def _dense_setup_prototype(self, *args, **kwargs):
        # Collect flat and individual and aggregated flat shapes.
        self._dense_shapes = {}
        pos = 0
        offsets = {}
        for d, event_shape in self._unconstrained_event_shapes.items():
            batch_shape = self.prototype_trace.nodes[d]["fn"].batch_shape
            self._dense_shapes[d] = batch_shape, event_shape
            offsets[d] = pos
            pos += (batch_shape + event_shape).numel()
        self._dense_size = pos

        # Create sparse -> dense precision matrix maps.
        self._dense_factor_scatter = {}
        for d, site in self._sorted_sites.items():
            # Order inputs as in the model, so as to maximize sparsity of the
            # lower Cholesky parametrization of the precision matrix.
            event_indices = []
            for f in site["cond_indep_stack"]:
                if f.vectorized:
                    if f.name not in self._broken_plates[d]:
                        f.size
            for u in self.dependencies[d]:
                start = offsets[u]
                stop = start + self._broken_event_shapes[u].numel()
                event_indices.append(torch.arange(start, stop))
            if not site["is_observed"]:
                start = offsets[d]
                stop = start + self._unconstrained_event_shapes[d].numel()
                event_indices.append(torch.arange(start, stop))
            event_index = torch.cat(event_indices)
            precision_shape = deep_getattr(self.precision_chols, d).shape
            index = torch.zeros(precision_shape, dtype=torch.long)
            stride = 1
            index += event_index * stride
            stride *= self._dense_size
            index += event_index[:, None] * stride
            stride *= self._dense_size
            # TODO add batch shapes
            self._dense_factor_scatter[d] = index.reshape(-1)

    def _dense_sample_aux_values(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Union[float, torch.Tensor]]:
        from pyro.ops import Gaussian

        # Convert to a flat dense joint Gaussian.
        flat_precision = torch.zeros(self._dense_size ** 2)
        for d, index in self._dense_factor_scatter:
            precision_chol = deep_getattr(self.precision_chols, d)
            precision = precision_chol @ precision_chol.transpose(-1, -2)
            flat_precision.scatter_add_(0, index, precision.reshape(-1))
        precision = flat_precision.reshape(self._dense_size, self._dense_size)
        info_vec = torch.zeros(self._dense_size)
        log_normalizer = torch.zeros(())
        g = Gaussian(log_normalizer, info_vec, precision)

        # Draw a batch of samples.
        particle_plates = frozenset(get_plates())
        sample_shape = [1] * max([0] + [-p.dim for p in particle_plates])
        for p in particle_plates:
            sample_shape[p.dim] = p.size
        sample_shape = torch.Size(sample_shape)
        flat_samples = g.rsample(sample_shape)
        log_density = g.log_density(flat_samples) - g.event_logsumexp()

        # Convert flat to shaped tensors.
        samples = {}
        pos = 0
        for d, (batch_shape, event_shape) in self._dense_shapes.items():
            numel = deep_getattr(self.locs, d).numel()
            flat_sample = flat_samples[pos : pos + numel]
            pos += numel
            # Assumes sample shapes are left of batch shapes.
            samples[d] = flat_sample.reshape(
                torch.broadcast_shapes(sample_shape, batch_shape) + event_shape
            )
        return samples, log_density

    ############################################################################
    # Funsor backend

    def _funsor_setup_prototype(self, *args, **kwargs):
        try:
            import funsor
        except ImportError as e:
            raise ImportError(
                'AutoGaussian(..., backend="funsor") requires funsor. '
                "Try installing via: pip install pyro-ppl[funsor]"
            ) from e
        funsor.set_backend("torch")

        # Determine TVE problem shape.
        factor_inputs: Dict[str, OrderedDict[str, funsor.Domain]] = {}
        eliminate: Set[str] = set()
        plate_to_dim: Dict[str, int] = {}

        for d, site in self._sorted_sites.items():
            # Order inputs as in the model, so as to maximize sparsity of the
            # lower Cholesky parametrization of the precision matrix.
            inputs = OrderedDict()
            for f in site["cond_indep_stack"]:
                if f.vectorized:
                    plate_to_dim[f.name] = f.dim
                    if f.name not in self._broken_plates[d]:
                        inputs[f.name] = funsor.Bint[f.size]
                        eliminate.add(f.name)
            for u in self.dependencies[d]:
                inputs[u] = funsor.Reals[self._broken_event_shapes[u]]
                eliminate.add(u)
            if not site["is_observed"]:
                inputs[d] = funsor.Reals[self._broken_event_shapes[d]]
                assert d in eliminate
            factor_inputs[d] = inputs

        self._funsor_factor_inputs = factor_inputs
        self._funsor_eliminate = frozenset(eliminate)
        self._funsor_plate_to_dim = plate_to_dim
        self._funsor_plates = frozenset(plate_to_dim)

    def _funsor_sample_aux_values(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Union[float, torch.Tensor]]:
        import funsor

        # Convert torch to funsor.
        particle_plates = frozenset(get_plates())
        plate_to_dim = self._funsor_plate_to_dim.copy()
        plate_to_dim.update({f.name: f.dim for f in particle_plates})
        factors = {}
        for d, inputs in self._funsor_factor_inputs.items():
            precision_chol = deep_getattr(self.precision_chols, d)
            precision = precision_chol @ precision_chol.transpose(-1, -2)
            info_vec = precision.new_zeros(()).expand(precision.shape[:-1])
            factors[d] = funsor.gaussian.Gaussian(info_vec, precision, inputs)
            factors[d]._precision_chol = precision_chol  # avoid recomputing

        # Perform Gaussian tensor variable elimination.
        samples, log_prob = funsor.recipes.forward_filter_backward_rsample(
            factors=factors,
            eliminate=self._funsor_eliminate,
            plates=frozenset(plate_to_dim),
            sample_inputs={f.name: funsor.Bint[f.size] for f in particle_plates},
        )

        # Convert funsor to torch.
        samples = {
            k: funsor.to_data(v[self._broken_plates[k]], name_to_dim=plate_to_dim)
            for k, v in samples.items()
        }
        log_density = funsor.to_data(log_prob, name_to_dim=plate_to_dim)

        return samples, log_density
