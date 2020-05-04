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
        factors, measures, measure_vars, plate_vars = [], [], frozenset(), frozenset()
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
        guide_aux_vars = frozenset().union(*(f.inputs for f in guide_log_factors)) - \
            frozenset(guide_plates) - \
            frozenset(model_log_factors)  # TODO get this right
        if guide_aux_vars:
            guide_log_probs = funsor.sum_product.partial_sum_product(
                funsor.ops.logaddexp, funsor.ops.add, guide_log_probs,
                plates=frozenset(guide_plates), eliminate=guide_aux_vars
            )

        # contract out auxiliary variables in the model
        model_aux_vars = frozenset().union(*(f.inputs for f in model_log_factors)) - \
            frozenset(model_plates) - \
            frozenset(guide_log_factors)  # TODO get this right
        if model_aux_vars:
            model_log_probs = funsor.sum_product.partial_sum_product(
                funsor.ops.logaddexp, funsor.ops.add, model_log_probs,
                plates=frozenset(model_plates), eliminate=model_aux_vars
            )

        # compute remaining plates and sum_dims
        plates = frozenset().union(
            *(model_plates.intersection(f.inputs) for f in model_log_probs))
        plates = plates | frozenset().union(
            *(guide_plates.intersection(f.inputs) for f in guide_log_probs))
        # TODO get sum_vars right
        sum_vars = frozenset().union(model_log_probs, guide_log_probs) - \
            frozenset(model_aux_vars | guide_aux_vars)

        # TODO inline this final bit
        with funsor.interpreter.interpretation(funsor.terms.lazy):
            elbo = Expectation(tuple(log_probs),  # TODO define args correctly
                               tuple(costs),
                               sum_vars=sum_vars,
                               prod_vars=plates)

        with funsor.memoize.memoize():
            return -to_data(funsor.optimizer.apply_optimizer(elbo))
