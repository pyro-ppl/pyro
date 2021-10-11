# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, plate, replay, trace
from pyro.contrib.funsor.infer.elbo import ELBO, Jit_ELBO
from pyro.distributions.util import copy_docs_from
from pyro.infer import TraceEnum_ELBO as _OrigTraceEnum_ELBO

from .trace_elbo import Trace_ELBO, apply_optimizer, terms_from_trace


@copy_docs_from(_OrigTraceEnum_ELBO)
class TraceMarkovEnum_ELBO(ELBO):
    def differentiable_loss(self, model, guide, *args, **kwargs):

        # get batched, enumerated, to_funsor-ed traces from the guide and model
        with plate(
            size=self.num_particles
        ) if self.num_particles > 1 else contextlib.ExitStack(), enum(
            first_available_dim=(-self.max_plate_nesting - 1)
            if self.max_plate_nesting
            else None
        ):
            guide_tr = trace(guide).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        # extract from traces all metadata that we will need to compute the elbo
        guide_terms = terms_from_trace(guide_tr)
        model_terms = terms_from_trace(model_tr)

        # guide side enumeration is not supported
        if any(guide_terms["plate_to_step"].values()):
            raise NotImplementedError(
                "TraceMarkovEnum_ELBO does not yet support guide side Markov enumeration"
            )

        # build up a lazy expression for the elbo
        with funsor.terms.lazy:
            # identify and contract out auxiliary variables in the model with partial_sum_product
            contracted_factors, uncontracted_factors = [], []
            for f in model_terms["log_factors"]:
                if model_terms["measure_vars"].intersection(f.inputs):
                    contracted_factors.append(f)
                else:
                    uncontracted_factors.append(f)
            # incorporate the effects of subsampling and handlers.scale through a common scale factor
            markov_dims = frozenset(
                {plate for plate, step in model_terms["plate_to_step"].items() if step}
            )
            contracted_costs = [
                model_terms["scale"] * f
                for f in funsor.sum_product.dynamic_partial_sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    model_terms["log_measures"] + contracted_factors,
                    plate_to_step=model_terms["plate_to_step"],
                    eliminate=model_terms["measure_vars"] | markov_dims,
                )
            ]

            costs = contracted_costs + uncontracted_factors  # model costs: logp
            costs += [-f for f in guide_terms["log_factors"]]  # guide costs: -logq

            # finally, integrate out guide variables in the elbo and all plates
            plate_vars = guide_terms["plate_vars"] | model_terms["plate_vars"]
            elbo = to_funsor(0, output=funsor.Real)
            for cost in costs:
                # compute the marginal logq in the guide corresponding to this cost term
                log_prob = funsor.sum_product.sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    guide_terms["log_measures"],
                    plates=plate_vars,
                    eliminate=(plate_vars | guide_terms["measure_vars"])
                    - frozenset(cost.inputs),
                )
                # compute the expected cost term E_q[logp] or E_q[-logq] using the marginal logq for q
                elbo_term = funsor.Integrate(
                    log_prob, cost, guide_terms["measure_vars"] & frozenset(cost.inputs)
                )
                elbo += elbo_term.reduce(
                    funsor.ops.add, plate_vars & frozenset(cost.inputs)
                )

        # evaluate the elbo, using memoize to share tensor computation where possible
        with funsor.interpretations.memoize():
            return -to_data(apply_optimizer(elbo))


class TraceEnum_ELBO(Trace_ELBO):
    pass


class JitTraceEnum_ELBO(Jit_ELBO, TraceEnum_ELBO):
    pass


class JitTraceMarkovEnum_ELBO(Jit_ELBO, TraceMarkovEnum_ELBO):
    pass
