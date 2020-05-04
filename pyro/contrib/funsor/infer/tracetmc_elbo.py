# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor

from pyro.infer import ELBO
from pyro.poutine.util import prune_subsample_sites

from pyro.contrib.funsor import to_data
from pyro.contrib.funsor.handlers import enum, replay, trace

funsor.set_backend("torch")


class TraceTMC_ELBO(ELBO):

    def _get_trace(self, *args, **kwargs):
        raise ValueError("shouldn't be here")

    def differentiable_loss(self, model, guide, *args, **kwargs):
        with trace() as guide_tr, enum(first_available_dim=-1-self.max_plate_nesting):
            guide(*args, **kwargs)
        with trace() as model_tr, enum(first_available_dim=-1-self.max_plate_nesting), \
                replay(trace=guide_tr.trace):
            model(*args, **kwargs)
        factors, measures, sum_vars, plate_vars = [], [], frozenset(), frozenset()
        for role, tr in zip(("model", "guide"), (model_tr.trace, guide_tr.trace)):
            tr = prune_subsample_sites(tr)
            for name, node in tr.nodes.items():
                if node["type"] != "sample":
                    continue
                if role == "model":
                    factors.append(node["funsor"]["log_prob"])
                    if name not in guide_tr.trace.nodes and not node["is_observed"]:
                        measures.append(node["funsor"]["log_measure"])
                        sum_vars |= frozenset(node["funsor"]["log_measure"].inputs)
                elif role == "guide":
                    factors.append(-node["funsor"]["log_prob"])
                    measures.append(node["funsor"]["log_measure"])
                    sum_vars |= frozenset(node["funsor"]["log_measure"].inputs)
                plate_vars |= frozenset(f.name for f in node["cond_indep_stack"] if f.vectorized)
                sum_vars |= frozenset(node["funsor"]["log_prob"].inputs)
                sum_vars -= plate_vars

        with funsor.interpreter.interpretation(funsor.terms.lazy):
            elbo = funsor.sum_product.sum_product(
                funsor.ops.logaddexp, funsor.ops.add,
                measures + factors, eliminate=sum_vars | plate_vars, plates=plate_vars
            )
        return -to_data(funsor.optimizer.apply_optimizer(elbo))
