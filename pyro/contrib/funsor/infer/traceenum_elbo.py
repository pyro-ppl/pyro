# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor
from funsor.adjoint import AdjointTape
from funsor.sum_product import _partition

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, plate, replay, trace
from pyro.contrib.funsor.infer.elbo import ELBO, Jit_ELBO
from pyro.distributions.util import copy_docs_from
from pyro.infer import TraceEnum_ELBO as _OrigTraceEnum_ELBO


# Work around a bug in unfold_contraction_generic_tuple interacting with
# Approximate introduced in https://github.com/pyro-ppl/funsor/pull/488 .
# Once fixed, this can be replaced by funsor.optimizer.apply_optimizer().
def apply_optimizer(x):
    with funsor.interpretations.normalize:
        expr = funsor.interpreter.reinterpret(x)

    with funsor.optimizer.optimize_base:
        return funsor.interpreter.reinterpret(expr)


def terms_from_trace(tr):
    """Helper function to extract elbo components from execution traces."""
    # data structure containing densities, measures, scales, and identification
    # of free variables as either product (plate) variables or sum (measure) variables
    terms = {
        "log_factors": [],
        "log_measures": [],
        "scale": to_funsor(1.0),
        "plate_vars": frozenset(),
        "measure_vars": frozenset(),
        "plate_to_step": dict(),
    }
    for name, node in tr.nodes.items():
        # add markov dimensions to the plate_to_step dictionary
        if node["type"] == "markov_chain":
            terms["plate_to_step"][node["name"]] = node["value"]
            # ensure previous step variables are added to measure_vars
            for step in node["value"]:
                terms["measure_vars"] |= frozenset(
                    {
                        var
                        for var in step[1:-1]
                        if tr.nodes[var]["funsor"].get("log_measure", None) is not None
                    }
                )
        if (
            node["type"] != "sample"
            or type(node["fn"]).__name__ == "_Subsample"
            or node["infer"].get("_do_not_score", False)
        ):
            continue
        # grab plate dimensions from the cond_indep_stack
        terms["plate_vars"] |= frozenset(
            f.name for f in node["cond_indep_stack"] if f.vectorized
        )
        # grab the log-measure, found only at sites that are not replayed or observed
        if node["funsor"].get("log_measure", None) is not None:
            terms["log_measures"].append(node["funsor"]["log_measure"])
            # sum (measure) variables: the fresh non-plate variables at a site
            terms["measure_vars"] |= (
                frozenset(node["funsor"]["value"].inputs) | {name}
            ) - terms["plate_vars"]
        # grab the scale, assuming a common subsampling scale
        if (
            node.get("replay_active", False)
            and set(node["funsor"]["log_prob"].inputs) & terms["measure_vars"]
            and float(to_data(node["funsor"]["scale"])) != 1.0
        ):
            # model site that depends on enumerated variable: common scale
            terms["scale"] = node["funsor"]["scale"]
        else:  # otherwise: default scale behavior
            node["funsor"]["log_prob"] = (
                node["funsor"]["log_prob"] * node["funsor"]["scale"]
            )
        # grab the log-density, found at all sites except those that are not replayed
        if node["is_observed"] or not node.get("replay_skipped", False):
            terms["log_factors"].append(node["funsor"]["log_prob"])
    # add plate dimensions to the plate_to_step dictionary
    terms["plate_to_step"].update(
        {plate: terms["plate_to_step"].get(plate, {}) for plate in terms["plate_vars"]}
    )
    return terms


@copy_docs_from(_OrigTraceEnum_ELBO)
class TraceMarkovEnum_ELBO(ELBO):
    def differentiable_loss(self, model, guide, *args, **kwargs):
        # get batched, enumerated, to_funsor-ed traces from the guide and model
        with (
            plate(size=self.num_particles)
            if self.num_particles > 1
            else contextlib.ExitStack()
        ), enum(
            first_available_dim=(
                (-self.max_plate_nesting - 1) if self.max_plate_nesting else None
            )
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


@copy_docs_from(_OrigTraceEnum_ELBO)
class TraceEnum_ELBO(ELBO):
    def differentiable_loss(self, model, guide, *args, **kwargs):
        # get batched, enumerated, to_funsor-ed traces from the guide and model
        with (
            plate(size=self.num_particles)
            if self.num_particles > 1
            else contextlib.ExitStack()
        ), enum(
            first_available_dim=(
                (-self.max_plate_nesting - 1) if self.max_plate_nesting else None
            )
        ):
            guide_tr = trace(guide).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        # extract from traces all metadata that we will need to compute the elbo
        guide_terms = terms_from_trace(guide_tr)
        model_terms = terms_from_trace(model_tr)

        # build up a lazy expression for the elbo
        with funsor.terms.lazy:
            # identify and contract out auxiliary variables in the model with partial_sum_product
            contracted_factors, uncontracted_factors = [], []
            for f in model_terms["log_factors"]:
                if model_terms["measure_vars"].intersection(f.inputs):
                    contracted_factors.append(f)
                else:
                    uncontracted_factors.append(f)
            contracted_costs = []
            # incorporate the effects of subsampling and handlers.scale through a common scale factor
            for group_factors, group_vars in _partition(
                model_terms["log_measures"] + contracted_factors,
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

            # compute expected cost
            # Cf. pyro.infer.util.Dice.compute_expectation()
            # https://github.com/pyro-ppl/pyro/blob/0.3.0/pyro/infer/util.py#L212
            # TODO Replace this with funsor.Expectation
            plate_vars = guide_terms["plate_vars"] | model_terms["plate_vars"]
            # compute the marginal logq in the guide corresponding to each cost term
            targets = dict()
            for cost in costs:
                input_vars = frozenset(cost.inputs)
                if input_vars not in targets:
                    targets[input_vars] = funsor.Tensor(
                        funsor.ops.new_zeros(
                            funsor.tensor.get_default_prototype(),
                            tuple(v.size for v in cost.inputs.values()),
                        ),
                        cost.inputs,
                        cost.dtype,
                    )
            with AdjointTape() as tape:
                logzq = funsor.sum_product.sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    guide_terms["log_measures"] + list(targets.values()),
                    plates=plate_vars,
                    eliminate=(plate_vars | guide_terms["measure_vars"]),
                )
            marginals = tape.adjoint(
                funsor.ops.logaddexp, funsor.ops.add, logzq, tuple(targets.values())
            )
            # finally, integrate out guide variables in the elbo and all plates
            elbo = to_funsor(0, output=funsor.Real)
            for cost in costs:
                target = targets[frozenset(cost.inputs)]
                logzq_local = marginals[target].reduce(
                    funsor.ops.logaddexp, frozenset(cost.inputs) - plate_vars
                )
                log_prob = marginals[target] - logzq_local
                elbo_term = funsor.Integrate(
                    log_prob,
                    cost,
                    guide_terms["measure_vars"] & frozenset(log_prob.inputs),
                )
                elbo += elbo_term.reduce(
                    funsor.ops.add, plate_vars & frozenset(cost.inputs)
                )

        # evaluate the elbo, using memoize to share tensor computation where possible
        with funsor.interpretations.memoize():
            return -to_data(apply_optimizer(elbo))


class JitTraceEnum_ELBO(Jit_ELBO, TraceEnum_ELBO):
    pass


class JitTraceMarkovEnum_ELBO(Jit_ELBO, TraceMarkovEnum_ELBO):
    pass
