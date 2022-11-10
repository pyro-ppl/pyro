# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor
from funsor.sum_product import _partition

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, plate, provenance, replay, trace
from pyro.distributions.util import copy_docs_from
from pyro.infer import Trace_ELBO as _OrigTrace_ELBO

from .elbo import ELBO, Jit_ELBO
from .traceenum_elbo import terms_from_trace


@copy_docs_from(_OrigTrace_ELBO)
class Trace_ELBO(ELBO):
    def differentiable_loss(self, model, guide, *args, **kwargs):
        with enum(
            first_available_dim=(-self.max_plate_nesting - 1)
            if self.max_plate_nesting is not None
            and self.max_plate_nesting != float("inf")
            else None
        ), provenance(), plate(
            name="num_particles_vectorized",
            size=self.num_particles,
            dim=-self.max_plate_nesting,
        ) if self.num_particles > 1 else contextlib.ExitStack():
            guide_tr = trace(guide).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        model_terms = terms_from_trace(model_tr)
        guide_terms = terms_from_trace(guide_tr)

        particle_var = (
            frozenset({"num_particles_vectorized"})
            if self.num_particles > 1
            else frozenset()
        )
        plate_vars = (
            guide_terms["plate_vars"] | model_terms["plate_vars"]
        ) - particle_var

        model_measure_vars = model_terms["measure_vars"] - guide_terms["measure_vars"]
        with funsor.terms.lazy:
            # identify and contract out auxiliary variables in the model with partial_sum_product
            contracted_factors, uncontracted_factors = [], []
            for f in model_terms["log_factors"]:
                if model_measure_vars.intersection(f.inputs):
                    contracted_factors.append(f)
                else:
                    uncontracted_factors.append(f)
            contracted_costs = []
            # incorporate the effects of subsampling and handlers.scale through a common scale factor
            for group_factors, group_vars in _partition(
                list(model_terms["log_measures"].values()) + contracted_factors,
                model_terms["measure_vars"],
            ):
                group_factor_vars = frozenset().union(
                    *[f.inputs for f in group_factors]
                )
                group_plates = model_terms["plate_vars"] & group_factor_vars
                outermost_plates = frozenset.intersection(
                    *(frozenset(f.inputs) & group_plates for f in group_factors)
                )
                elim_plates = group_plates - outermost_plates
                for f in funsor.sum_product.partial_sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    group_factors,
                    plates=group_plates,
                    eliminate=group_vars | elim_plates,
                ):
                    contracted_costs.append(model_terms["scale"] * f)

            # accumulate costs from model (logp) and guide (-logq)
            costs = contracted_costs + uncontracted_factors  # model costs: logp
            costs += [-f for f in guide_terms["log_factors"]]  # guide costs: -logq

        # compute log_measures corresponding to each cost term
        # the goal is to achieve fine-grained Rao-Blackwellization
        log_measures = dict()
        for cost in costs:
            if cost.input_vars not in log_measures:
                log_probs = [
                    f
                    for name, f in guide_terms["log_measures"].items()
                    if name in cost.inputs
                ]
                log_prob = funsor.sum_product.sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    log_probs,
                    plates=plate_vars,
                    eliminate=(plate_vars | guide_terms["measure_vars"])
                    - frozenset(cost.inputs),
                )
                log_measures[cost.input_vars] = funsor.optimizer.apply_optimizer(
                    log_prob
                )

        with funsor.terms.lazy:
            # finally, integrate out guide variables in the elbo and all plates
            elbo = to_funsor(0, output=funsor.Real)
            for cost in costs:
                log_measure = log_measures[cost.input_vars]
                measure_vars = (frozenset(cost.inputs) - plate_vars) - particle_var
                elbo_term = funsor.Integrate(
                    log_measure,
                    cost,
                    measure_vars,
                )
                elbo += elbo_term.reduce(
                    funsor.ops.add, plate_vars & frozenset(cost.inputs)
                )
            # average over Monte-Carlo particles
            elbo = elbo.reduce(funsor.ops.mean, particle_var)

        return -to_data(funsor.optimizer.apply_optimizer(elbo))


class JitTrace_ELBO(Jit_ELBO, Trace_ELBO):
    pass
