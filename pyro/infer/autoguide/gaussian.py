# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from operator import attrgetter
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Set, Tuple, Union

import torch
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import constraints
from pyro.infer.inspect import get_dependencies, is_sample_site
from pyro.nn.module import PyroModule, PyroParam
from pyro.ops.linalg import ignore_torch_deprecation_warnings
from pyro.poutine.runtime import am_i_wrapped, get_plates
from pyro.poutine.util import site_is_subsample

from .guides import AutoGuide
from .initialization import InitMessenger, init_to_feasible
from .utils import deep_setattr, helpful_support_errors


# Helper to dispatch to concrete subclasses of AutoGaussian, e.g.
#   AutoGaussian(model, backend="dense")
# is converted to
#   AutoGaussianDense(model)
# The intent is to avoid proliferation of subclasses and docstrings,
# and provide a single interface AutoGaussian(...).
class AutoGaussianMeta(type(AutoGuide), ABCMeta):
    backends = {}
    default_backend = "dense"

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert cls.__name__.startswith("AutoGaussian")
        key = cls.__name__.replace("AutoGaussian", "").lower()
        cls.backends[key] = cls

    def __call__(cls, *args, **kwargs):
        if cls is AutoGaussian:
            backend = kwargs.pop("backend", cls.default_backend)
            cls = cls.backends[backend]
        return super(AutoGaussianMeta, cls).__call__(*args, **kwargs)


class AutoGaussian(AutoGuide, metaclass=AutoGaussianMeta):
    """
    Gaussian guide with optimal conditional independence structure.

    This is equivalent to a full rank :class:`AutoMultivariateNormal` guide,
    but with a sparse precision matrix determined by dependencies and plates in
    the model [1]. Depending on model structure, this can have asymptotically
    better statistical efficiency than :class:`AutoMultivariateNormal` .

    This guide implements multiple backends for computation. All backends use
    the same statistically optimal parametrization. The default "dense" backend
    has computational complexity similar to :class:`AutoMultivariateNormal` .
    The experimental "funsor" backend can be asymptotically cheaper in terms of
    time and space (using Gaussian tensor variable elimination [2,3]), but
    incurs large constant overhead. The "funsor" backend requires `funsor
    <https://funsor.pyro.ai>`_ which can be installed via ``pip install
    pyro-ppl[funsor]``.

    The guide currently does not depend on the model's ``*args, **kwargs``.

    Example::

        guide = AutoGaussian(model)
        svi = SVI(model, guide, ...)

    Example using experimental funsor backend::

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
    :param str backend: Back end for performing Gaussian tensor variable
        elimination. Defaults to "dense"; other options include "funsor".
    """

    scale_constraint = constraints.softplus_positive

    def __init__(
        self,
        model: Callable,
        *,
        init_loc_fn: Callable = init_to_feasible,
        init_scale: float = 0.1,
        backend: Optional[str] = None,  # used only by metaclass
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(f"Expected init_scale > 0. but got {init_scale}")
        self._init_scale = init_scale
        self._original_model = (model,)
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model)

    @staticmethod
    def _prototype_hide_fn(msg):
        # In contrast to the AutoGuide base class, this includes observation
        # sites and excludes deterministic sites.
        return not is_sample_site(msg)

    def _setup_prototype(self, *args, **kwargs) -> None:
        super()._setup_prototype(*args, **kwargs)

        self.locs = PyroModule()
        self.scales = PyroModule()
        self.white_vecs = PyroModule()
        self.prec_sqrts = PyroModule()
        self._factors = OrderedDict()
        self._plates = OrderedDict()
        self._event_numel = OrderedDict()
        self._unconstrained_event_shapes = OrderedDict()

        # Trace model dependencies.
        model = self._original_model[0]
        self._original_model = None
        self.dependencies = poutine.block(get_dependencies)(model, args, kwargs)[
            "prior_dependencies"
        ]

        # Eliminate observations with no upstream latents.
        for d, upstreams in list(self.dependencies.items()):
            if all(self.prototype_trace.nodes[u]["is_observed"] for u in upstreams):
                del self.dependencies[d]
                del self.prototype_trace.nodes[d]

        # Collect factors and plates.
        for d, site in self.prototype_trace.nodes.items():
            # Prune non-essential parts of the trace to save memory.
            pruned_site, site = site, site.copy()
            pruned_site.clear()

            # Collect factors and plates.
            if site["type"] != "sample" or site_is_subsample(site):
                continue
            assert all(f.vectorized for f in site["cond_indep_stack"])
            self._factors[d] = self._compress_site(site)
            plates = frozenset(site["cond_indep_stack"])
            if site["fn"].batch_shape != _plates_to_shape(plates):
                raise ValueError(
                    f"Shape mismatch at site '{d}'. "
                    "Are you missing a pyro.plate() or .to_event()?"
                )
            if site["is_observed"]:
                # Break irrelevant observation plates.
                plates &= frozenset().union(
                    *(self._plates[u] for u in self.dependencies[d] if u != d)
                )
            self._plates[d] = plates

            # Create location-scale parameters, one per latent variable.
            if site["is_observed"]:
                # This may slightly overestimate, e.g. for Multinomial.
                self._event_numel[d] = site["fn"].event_shape.numel()
                # Account for broken irrelevant observation plates.
                for f in set(site["cond_indep_stack"]) - plates:
                    self._event_numel[d] *= f.size
                continue
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
                    broken_shape = _plates_to_shape(self._plates[u] - self._plates[d])
                    u_size += broken_shape.numel() * self._event_numel[u]
            d_size = self._event_numel[d]
            if site["is_observed"]:
                d_size = min(d_size, u_size)  # just an optimization
            batch_shape = _plates_to_shape(self._plates[d])

            # Create parameters of each Gaussian factor.
            white_vec = init_loc.new_zeros(batch_shape + (d_size,))
            # We initialize with noise to avoid singular gradient.
            prec_sqrt = torch.rand(
                batch_shape + (u_size, d_size),
                dtype=init_loc.dtype,
                device=init_loc.device,
            )
            prec_sqrt.sub_(0.5).mul_(self._init_scale)
            if not site["is_observed"]:
                # Initialize the [d,d] block to the identity matrix.
                prec_sqrt.diagonal(dim1=-2, dim2=-1).fill_(1)
            deep_setattr(self.white_vecs, d, PyroParam(white_vec, event_dim=1))
            deep_setattr(self.prec_sqrts, d, PyroParam(prec_sqrt, event_dim=2))

    @staticmethod
    def _compress_site(site):
        # Save memory by retaining only necessary parts of the site.
        return {
            "name": site["name"],
            "type": site["type"],
            "cond_indep_stack": site["cond_indep_stack"],
            "is_observed": site["is_observed"],
            "fn": SimpleNamespace(
                support=site["fn"].support,
                batch_shape=site["fn"].batch_shape,
                event_dim=site["fn"].event_dim,
            ),
        }

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        aux_values = self._sample_aux_values(temperature=1.0)
        values, log_densities = self._transform_values(aux_values)

        # Replay via Pyro primitives.
        plates = self._create_plates(*args, **kwargs)
        for name, site in self._factors.items():
            if site["is_observed"]:
                continue
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])
                values[name] = pyro.sample(
                    name,
                    dist.Delta(values[name], log_densities[name], site["fn"].event_dim),
                )
        return values

    def median(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        with torch.no_grad(), poutine.mask(mask=False):
            aux_values = self._sample_aux_values(temperature=0.0)
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
            if site["is_observed"]:
                continue
            loc = attrgetter(name)(self.locs)
            scale = attrgetter(name)(self.scales)
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

    @abstractmethod
    def _sample_aux_values(self, *, temperature: float) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class AutoGaussianDense(AutoGaussian):
    """
    Dense implementation of :class:`AutoGaussian` .

    The following are equivalent::

        guide = AutoGaussian(model, backend="dense")
        guide = AutoGaussianDense(model)
    """

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        # Collect global shapes and per-axis indices.
        self._dense_shapes = {}
        global_indices = {}
        pos = 0
        for d, event_shape in self._unconstrained_event_shapes.items():
            batch_shape = self._factors[d]["fn"].batch_shape
            self._dense_shapes[d] = batch_shape, event_shape
            end = pos + (batch_shape + event_shape).numel()
            global_indices[d] = torch.arange(pos, end).reshape(batch_shape + (-1,))
            pos = end
        self._dense_size = pos

        # Create sparse -> dense precision scatter indices.
        self._dense_scatter = {}
        for d, site in self._factors.items():
            prec_sqrt_shape = attrgetter(d)(self.prec_sqrts).shape
            info_vec_shape = prec_sqrt_shape[:-1]
            precision_shape = prec_sqrt_shape[:-1] + prec_sqrt_shape[-2:-1]
            index1 = torch.zeros(info_vec_shape, dtype=torch.long)
            index2 = torch.zeros(precision_shape, dtype=torch.long)

            # Collect local offsets and create index1 for info_vec blockwise.
            upstreams = [
                u for u in self.dependencies[d] if not self._factors[u]["is_observed"]
            ]
            local_offsets = {}
            pos = 0
            for u in upstreams:
                local_offsets[u] = pos
                broken_plates = self._plates[u] - self._plates[d]
                pos += self._event_numel[u] * _plates_to_shape(broken_plates).numel()
                u_index = global_indices[u]

                # Permute broken plates to the right of preserved plates.
                u_index = _break_plates(u_index, self._plates[u], self._plates[d])

                # Scatter global indices into the [u] block.
                u_start = local_offsets[u]
                u_stop = u_start + u_index.size(-1)
                index1[..., u_start:u_stop] = u_index

            # Create index2 for precision blockwise.
            for u, v in itertools.product(upstreams, upstreams):
                u_index = global_indices[u]
                v_index = global_indices[v]

                # Permute broken plates to the right of preserved plates.
                u_index = _break_plates(u_index, self._plates[u], self._plates[d])
                v_index = _break_plates(v_index, self._plates[v], self._plates[d])

                # Scatter global indices into the [u,v] block.
                u_start = local_offsets[u]
                u_stop = u_start + u_index.size(-1)
                v_start = local_offsets[v]
                v_stop = v_start + v_index.size(-1)
                index2[..., u_start:u_stop, v_start:v_stop] = (
                    self._dense_size * u_index.unsqueeze(-1) + v_index.unsqueeze(-2)
                )

            self._dense_scatter[d] = index1.reshape(-1), index2.reshape(-1)

    def _sample_aux_values(self, *, temperature: float) -> Dict[str, torch.Tensor]:
        mvn = self._dense_get_mvn()
        if temperature == 0:
            # Simply return the mode.
            flat_samples = mvn.mean
        elif temperature == 1:
            # Sample from a dense joint Gaussian over flattened variables.
            flat_samples = pyro.sample(
                f"_{self._pyro_name}_latent", mvn, infer={"is_auxiliary": True}
            )
        else:
            raise NotImplementedError(f"Invalid temperature: {temperature}")
        samples = self._dense_unflatten(flat_samples)
        return samples

    def _dense_unflatten(self, flat_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Convert a single flattened sample to a dict of shaped samples.
        sample_shape = flat_samples.shape[:-1]
        samples = {}
        pos = 0
        for d, (batch_shape, event_shape) in self._dense_shapes.items():
            end = pos + (batch_shape + event_shape).numel()
            flat_sample = flat_samples[..., pos:end]
            pos = end
            # Assumes sample shapes are left of batch shapes.
            samples[d] = flat_sample.reshape(
                torch.broadcast_shapes(sample_shape, batch_shape) + event_shape
            )
        return samples

    def _dense_flatten(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Convert a dict of shaped samples single flattened sample.
        flat_samples = []
        for d, (batch_shape, event_shape) in self._dense_shapes.items():
            shape = samples[d].shape
            sample_shape = shape[: len(shape) - len(batch_shape) - len(event_shape)]
            flat_samples.append(samples[d].reshape(sample_shape + (-1,)))
        return torch.cat(flat_samples, dim=-1)

    def _dense_get_mvn(self):
        # Create a dense joint Gaussian over flattened variables.
        flat_info_vec = torch.zeros(self._dense_size)
        flat_precision = torch.zeros(self._dense_size**2)
        for d, (index1, index2) in self._dense_scatter.items():
            white_vec = attrgetter(d)(self.white_vecs)
            prec_sqrt = attrgetter(d)(self.prec_sqrts)
            info_vec = (prec_sqrt @ white_vec[..., None])[..., 0]
            precision = prec_sqrt @ prec_sqrt.transpose(-1, -2)
            flat_info_vec.scatter_add_(0, index1, info_vec.reshape(-1))
            flat_precision.scatter_add_(0, index2, precision.reshape(-1))
        info_vec = flat_info_vec
        precision = flat_precision.reshape(self._dense_size, self._dense_size)
        scale_tril = _precision_to_scale_tril(precision)
        loc = (
            scale_tril @ (scale_tril.transpose(-1, -2) @ info_vec.unsqueeze(-1))
        ).squeeze(-1)
        return dist.MultivariateNormal(loc, scale_tril=scale_tril)


class AutoGaussianFunsor(AutoGaussian):
    """
    Funsor implementation of :class:`AutoGaussian` .

    The following are equivalent::
        guide = AutoGaussian(model, backend="funsor")
        guide = AutoGaussianFunsor(model)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _import_funsor()

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        funsor = _import_funsor()

        # Check TVE condition 1: plate nesting is monotone.
        for d in self._factors:
            pd = {p.name for p in self._plates[d]}
            for u in self.dependencies[d]:
                pu = {p.name for p in self._plates[u]}
                if pu <= pd:
                    continue  # ok
                raise NotImplementedError(
                    "Expected monotone plate nesting, but found dependency "
                    f"{repr(u)} -> {repr(d)} leaves plates {pu - pd}. "
                    "Consider splitting into multiple guides via AutoGuideList, "
                    "or replacing the plate in the model by .to_event()."
                )

        # Determine TVE problem shape.
        factor_inputs: Dict[str, OrderedDict[str, funsor.Domain]] = {}
        eliminate: Set[str] = set()
        plate_to_dim: Dict[str, int] = {}
        for d, site in self._factors.items():
            inputs = OrderedDict()
            for f in sorted(self._plates[d], key=lambda f: f.dim):
                plate_to_dim[f.name] = f.dim
                inputs[f.name] = funsor.Bint[f.size]
                eliminate.add(f.name)
            for u in self.dependencies[d]:
                if self._factors[u]["is_observed"]:
                    continue
                inputs[u] = funsor.Reals[self._unconstrained_event_shapes[u]]
                eliminate.add(u)
            factor_inputs[d] = inputs

        self._funsor_factor_inputs = factor_inputs
        self._funsor_eliminate = frozenset(eliminate)
        self._funsor_plate_to_dim = plate_to_dim
        self._funsor_plates = frozenset(plate_to_dim)

    def _sample_aux_values(self, *, temperature: float) -> Dict[str, torch.Tensor]:
        funsor = _import_funsor()

        # Convert torch to funsor.
        particle_plates = frozenset(get_plates())
        plate_to_dim = self._funsor_plate_to_dim.copy()
        plate_to_dim.update({f.name: f.dim for f in particle_plates})
        factors = {}
        for d, inputs in self._funsor_factor_inputs.items():
            batch_shape = torch.Size(
                p.size for p in sorted(self._plates[d], key=lambda p: p.dim)
            )
            white_vec = attrgetter(d)(self.white_vecs)
            prec_sqrt = attrgetter(d)(self.prec_sqrts)
            factors[d] = funsor.gaussian.Gaussian(
                white_vec=white_vec.reshape(batch_shape + white_vec.shape[-1:]),
                prec_sqrt=prec_sqrt.reshape(batch_shape + prec_sqrt.shape[-2:]),
                inputs=inputs,
            )

        # Perform Gaussian tensor variable elimination.
        if temperature == 1:
            samples, log_prob = _try_possibly_intractable(
                funsor.recipes.forward_filter_backward_rsample,
                factors=factors,
                eliminate=self._funsor_eliminate,
                plates=frozenset(plate_to_dim),
                sample_inputs={f.name: funsor.Bint[f.size] for f in particle_plates},
            )

        else:
            samples, log_prob = _try_possibly_intractable(
                funsor.recipes.forward_filter_backward_precondition,
                factors=factors,
                eliminate=self._funsor_eliminate,
                plates=frozenset(plate_to_dim),
            )

            # Substitute noise.
            sample_shape = torch.Size(f.size for f in particle_plates)
            noise = torch.randn(sample_shape + log_prob.inputs["aux"].shape)
            noise.mul_(temperature)
            aux = funsor.Tensor(noise)[tuple(f.name for f in particle_plates)]
            with funsor.interpretations.memoize():
                samples = {k: v(aux=aux) for k, v in samples.items()}
                log_prob = log_prob(aux=aux)

        # Convert funsor to torch.
        if am_i_wrapped() and poutine.get_mask() is not False:
            log_prob = funsor.to_data(log_prob, name_to_dim=plate_to_dim)
            pyro.factor(f"_{self._pyro_name}_latent", log_prob, has_rsample=True)
        samples = {
            k: funsor.to_data(v, name_to_dim=plate_to_dim) for k, v in samples.items()
        }
        return samples


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.linalg.solve_triangular(
        L_inv, torch.eye(P.shape[-1], dtype=P.dtype, device=P.device), upper=False
    )
    return L


@ignore_torch_deprecation_warnings()
def _try_possibly_intractable(fn, *args, **kwargs):
    # Convert ValueError into NotImplementedError.
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        if str(e) != "intractable!":
            raise e from None
        raise NotImplementedError(
            "Funsor backend found intractable plate nesting. "
            'Consider using AutoGaussian(..., backend="dense"), '
            "splitting into multiple guides via AutoGuideList, or "
            "replacing some plates in the model by .to_event()."
        ) from e


def _plates_to_shape(plates):
    shape = [1] * max([0] + [-f.dim for f in plates])
    for f in plates:
        shape[f.dim] = f.size
    return torch.Size(shape)


def _break_plates(x, all_plates, kept_plates):
    """
    Reshapes and permutes a tensor ``x`` with event_dim=1 and batch shape given
    by ``all_plates`` by breaking all plates not in ``kept_plates``. Each
    broken plate is moved into the event shape, and finally the event shape is
    flattend back to a single dimension.
    """
    assert x.shape[:-1] == _plates_to_shape(all_plates)  # event_dim == 1
    kept_plates = kept_plates & all_plates
    broken_plates = all_plates - kept_plates

    if not broken_plates:
        return x

    if not kept_plates:
        # Empty batch shape.
        return x.reshape(-1)

    batch_shape = _plates_to_shape(kept_plates)
    if max(p.dim for p in kept_plates) < min(p.dim for p in broken_plates):
        # No permutation is necessary.
        return x.reshape(batch_shape + (-1,))

    # We need to permute broken plates left past kept plates.
    event_dims = {-1} | {p.dim - 1 for p in broken_plates}
    perm = sorted(range(-x.dim(), 0), key=lambda d: (d in event_dims, d))
    return x.permute(perm).reshape(batch_shape + (-1,))


def _import_funsor():
    try:
        import funsor
    except ImportError as e:
        raise ImportError(
            'AutoGaussian(..., backend="funsor") requires funsor. '
            "Try installing via: pip install pyro-ppl[funsor]"
        ) from e
    funsor.set_backend("torch")
    return funsor


__all__ = [
    "AutoGaussian",
]
