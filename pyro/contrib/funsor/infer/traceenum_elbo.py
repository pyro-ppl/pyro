# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor

from pyro.infer import ELBO
from pyro.poutine.util import prune_subsample_sites

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, replay, trace

funsor.set_backend("torch")


# TODO inline this
def Expectation(log_probs, costs, sum_vars, prod_vars):
    result = 0
    for cost in costs:
        log_prob = funsor.sum_product.sum_product(
            sum_op=funsor.ops.logaddexp,
            prod_op=funsor.ops.add,
            factors=log_probs,
            plates=prod_vars,
            eliminate=(prod_vars | sum_vars) - frozenset(cost.inputs)
        )
        term = funsor.Integrate(log_prob, cost, sum_vars & frozenset(cost.inputs))
        term = term.reduce(funsor.ops.add, prod_vars & frozenset(cost.inputs))
        result += term
    return result


class TraceEnum_ELBO(ELBO):

    def _get_trace(self, *args, **kwargs):
        raise ValueError("shouldn't be here")

    def differentiable_loss(self, model, guide, *args, **kwargs):
        with enum(first_available_dim=-self.max_plate_nesting-1):
            guide_tr = trace(guide).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        # TODO in this loop, compute factors, measures, plates, and elimination variables
        model_log_factors, model_log_measures, model_measure_vars, model_plate_vars = \
            [], [], frozenset(), frozenset()
        guide_log_factors, guide_log_measures, guide_measure_vars, guide_plate_vars = \
            [], [], frozenset(), frozenset()
        for role, tr in zip(("model", "guide"), (model_tr, guide_tr)):
            tr = prune_subsample_sites(tr)
            for name, node in tr.nodes.items():
                if role == "model":
                    factors.append(node["funsor"]["log_prob"])
                elif role == "guide":
                    factors.append(-node["funsor"]["log_prob"])
                    measures.append(node["funsor"]["log_measure"])
                plate_vars |= frozenset(f.name for f in node["cond_indep_stack"] if f.vectorized)
                measure_vars |= frozenset(node["funsor"]["log_prob"].inputs) - plate_vars

        # contract out auxiliary variables in the guide
        guide_aux_vars = guide_measure_vars - guide_plate_vars - (model_measure_vars | model_plate_vars)
        if guide_aux_vars:
            guide_log_factors = funsor.sum_product.partial_sum_product(
                funsor.ops.logaddexp, funsor.ops.add, guide_log_measures + guide_log_factors,
                plates=guide_plate_vars, eliminate=guide_aux_vars
            )

        # contract out auxiliary variables in the model
        model_aux_vars = model_measure_vars - model_plate_vars - (guide_measure_vars | guide_plate_vars)
        if model_aux_vars:
            model_log_factors = funsor.sum_product.partial_sum_product(
                funsor.ops.logaddexp, funsor.ops.add, model_log_measures + model_log_factors,
                plates=model_plate_vars, eliminate=model_aux_vars
            )

        # compute remaining plates and sum_dims
        plate_vars = (model_plate_vars | guide_plate_vars) - guide_aux_vars - model_aux_vars
        sum_vars = (model_measure_vars | guide_measure_vars) - guide_aux_vars - model_aux_vars - plate_vars

        # TODO inline this final bit
        with funsor.interpreter.interpretation(funsor.terms.lazy):
            elbo = Expectation(guide_log_factors, model_log_factors + [-lp for lp in guide_log_factors],
                               sum_vars, plate_vars)

        with funsor.memoize.memoize():
            return -to_data(funsor.optimizer.apply_optimizer(elbo))
