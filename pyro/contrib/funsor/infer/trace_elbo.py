# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor
import torch
from funsor.adjoint import AdjointTape
from funsor.constant import Constant

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, plate, provenance, replay, trace
from pyro.distributions.util import copy_docs_from
from pyro.infer import Trace_ELBO as _OrigTrace_ELBO

from .elbo import ELBO, Jit_ELBO
from .traceenum_elbo import apply_optimizer, terms_from_trace


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

        # identify and contract out auxiliary variables in the model with partial_sum_product
        contracted_factors, uncontracted_factors = [], []
        for f in model_terms["log_factors"]:
            if model_terms["measure_vars"].intersection(f.inputs):
                contracted_factors.append(f)
            else:
                uncontracted_factors.append(f)
        # incorporate the effects of subsampling and handlers.scale through a common scale factor
        contracted_costs = [
            model_terms["scale"] * f
            for f in funsor.sum_product.partial_sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                model_terms["log_measures"] + contracted_factors,
                plates=model_terms["plate_vars"],
                eliminate=model_terms["measure_vars"],
            )
        ]

        # accumulate costs from model (logp) and guide (-logq)
        costs = contracted_costs + uncontracted_factors  # model costs: logp
        costs += [-f for f in guide_terms["log_factors"]]  # guide costs: -logq

        plate_vars = (
            guide_terms["plate_vars"] | model_terms["plate_vars"]
        ) - frozenset({"num_particles_vectorized"})
        # compute log_measures corresponding to each cost term
        # the goal is to achieve fine-grained Rao-Blackwellization
        targets = dict()
        for cost in costs:
            if cost.input_vars not in targets:
                targets[cost.input_vars] = Constant(
                    cost.inputs, funsor.Tensor(torch.tensor(0.0))
                )
        with AdjointTape() as tape:
            logzq = funsor.sum_product.sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                guide_terms["log_measures"] + list(targets.values()),
                plates=plate_vars,
                eliminate=(plate_vars | guide_terms["measure_vars"]),
            )
        log_measures = tape.adjoint(
            funsor.ops.logaddexp, funsor.ops.add, logzq, tuple(targets.values())
        )
        with funsor.terms.eager:
            # finally, integrate out guide variables in the elbo and all plates
            elbo = to_funsor(0, output=funsor.Real)
            for cost in costs:
                target = targets[cost.input_vars]
                log_measure = log_measures[target]
                measure_vars = (frozenset(cost.inputs) - plate_vars) - frozenset(
                    {"num_particles_vectorized"}
                )
                elbo_term = funsor.Integrate(
                    log_measure,
                    cost,
                    measure_vars,
                )
                elbo += elbo_term.reduce(
                    funsor.ops.add, plate_vars & frozenset(cost.inputs)
                )
            elbo = elbo.reduce(funsor.ops.mean)

        # evaluate the elbo, using memoize to share tensor computation where possible
        with funsor.interpretations.memoize():
            return -to_data(apply_optimizer(elbo))


class JitTrace_ELBO(Jit_ELBO, Trace_ELBO):
    pass
