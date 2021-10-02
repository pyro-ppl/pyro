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

    default_backend: str = "dense"
    scale_constraint = constraints.softplus_positive

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
        self.factors = PyroModule()
        self._factors = OrderedDict()
        self._plates = OrderedDict()
        self._event_numel = OrderedDict()
        self._unconstrained_event_shapes = OrderedDict()

        # Trace model dependencies.
        model = self._original_model[0]
        self.dependencies = poutine.block(get_dependencies)(model, args, kwargs)[
            "prior_dependencies"
        ]

        # Collect factors and plates.
        for d, site in self.prototype_trace.nodes.items():
            if site["type"] == "sample" and not site_is_subsample(site):
                assert all(f.vectorized for f in site["cond_indep_stack"])
                self._factors[d] = site
                plates = frozenset(site["cond_indep_stack"])
                if site["is_observed"]:
                    # Eagerly eliminate irrelevant observation plates.
                    plates &= frozenset.union(
                        *(self._plates[u] for u in self.dependencies[d] if u != d)
                    )
                self._plates[d] = plates

        # Create location-scale parameters, one per latent variable.
        for d, site in self._factors.items():
            if not site["is_observed"]:
                with helpful_support_errors(site):
                    init_loc = biject_to(site["fn"].support).inv(site["value"]).detach()
                batch_shape = site["fn"].batch_shape
                event_shape = init_loc.shape[len(batch_shape) :]
                self._unconstrained_event_shapes[d] = event_shape
                self._event_numel[d] = event_shape.numel()
                event_dim = len(event_shape)
                deep_setattr(self.locs, d, PyroParam(init_loc, event_dim=event_dim))
                deep_setattr(
                    self.scales,
                    d,
                    PyroParam(
                        torch.full_like(init_loc, self._init_scale),
                        constraint=self.scale_constraint,
                        event_dim=event_dim,
                    ),
                )

        # Create parameters for dependencies, one per factor.
        for d, site in self._factors.items():
            u_size = 0
            for u in self.dependencies[d]:
                if not self._factors[u]["is_observed"]:
                    broken_shape = _plates_to_batch_shape(
                        self._plates[u] - self._plates[d]
                    )
                    u_size += broken_shape.numel() * self._event_numel[u]
            d_size = self._event_numel[d]
            if site["is_observed"]:
                d_size = min(d_size, u_size)  # just an optimization
            batch_shape = _plates_to_batch_shape(self._plates[d])

            # Create a square root (not necessarily lower triangular).
            raw = init_loc.new_zeros(batch_shape + (u_size, d_size))
            deep_setattr(self.factors, d, PyroParam(raw, event_dim=2))

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
        for name, site in self._factors.items():
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
            aux_values = {name: 0.0 for name in self._factors}
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
        for name, site in self._factors.items():
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
    # Dense backend. Methods and attributes are prefixed by ._dense_

    def _dense_setup_prototype(self, *args, **kwargs):
        # Collect flat and individual and aggregated flat shapes.
        self._dense_shapes = {}
        dense_gather = {}
        pos = 0
        for d, event_shape in self._unconstrained_event_shapes.items():
            batch_shape = self._factors[d]["fn"].batch_shape
            self._dense_shapes[d] = batch_shape, event_shape
            shape = batch_shape + event_shape
            end = pos + shape.numel()
            dense_gather[d] = torch.arange(pos, end).reshape(shape)
            pos = end
        self._dense_size = pos

        # Create sparse -> dense precision scatter indices.
        self._dense_scatter = {}
        for d, site in self._factors.items():
            raw_shape = deep_getattr(self.factors, d).shape
            precision_shape = raw_shape[:-1] + raw_shape[-2:-1]
            index = torch.zeros(precision_shape, dtype=torch.long)
            for u in self.dependencies[d]:
                if not self._factors[u]["is_observed"]:
                    batch_plates = self._plates[u] & self._plates[d]  # linear
                    broken_plates = self._plates[u] - self._plates[d]  # quadratic
                    event_numel = self._event_numel[u]  # quadratic
                    "TODO"
            self._dense_scatter[d] = index.reshape(-1)

        #########################################################
        # OLD
        for d, site in self._factors.items():
            event_indices = []
            for u in self.dependencies[d]:
                if not self._factors[u]["is_observed"]:
                    start = offsets[u]
                    stop = start + "TODO"
                    event_indices.append(torch.arange(start, stop))
            event_index = torch.cat(event_indices)
            raw_shape = deep_getattr(self.factors, d).shape
            precision_shape = raw_shape[:-1] + raw_shape[-2:-1]
            index = torch.zeros(precision_shape, dtype=torch.long)
            stride = 1
            index += event_index * stride
            stride *= self._dense_size
            index += event_index[:, None] * stride
            stride *= self._dense_size
            # TODO add batch shapes
            self._dense_scatter[d] = index.reshape(-1)

    def _dense_get_precision(self):
        flat_precision = torch.zeros(self._dense_size ** 2)
        for d, index in self._dense_scatter:
            raw = deep_getattr(self.factors, d)
            precision = _raw_to_precision(raw)
            flat_precision.scatter_add_(0, index, precision.reshape(-1))
        precision = flat_precision.reshape(self._dense_size, self._dense_size)
        return precision

    def _dense_sample_aux_values(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Union[float, torch.Tensor]]:
        from pyro.ops import Gaussian

        # Convert to a flat dense joint Gaussian.
        precision = self._dense_get_precision()
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
            end = pos + self._event_numel[d]
            flat_sample = flat_samples[pos:end]
            pos = end
            # Assumes sample shapes are left of batch shapes.
            samples[d] = flat_sample.reshape(
                torch.broadcast_shapes(sample_shape, batch_shape) + event_shape
            )
        return samples, log_density

    ############################################################################
    # Funsor backend. Methods and attributes are prefixed by ._funsor_

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

        for d, site in self._factors.items():
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


def _raw_to_precision(raw):
    """
    Transform an unconstrained matrix of shape ``batch_shape + (m, n)`` to a
    positive semidefinite precision matrix of shape ``batch_shape + (m, m)``.
    Typically ``m >= n``.
    """
    return raw @ raw.transpose(dim1=-2, dim2=-1)


def _plates_to_batch_shape(plates):
    shape = [1] * max([0] + [-f.dim for f in plates])
    for f in plates:
        shape[f.dim] = f.size
    return torch.Size(shape)
