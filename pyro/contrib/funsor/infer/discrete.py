# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import funsor

from pyro.contrib.funsor.handlers import enum, replay, trace
from pyro.contrib.funsor.handlers.enum_messenger import _get_support_value
from pyro.contrib.funsor.infer.traceenum_elbo import terms_from_trace
from pyro.poutine import block
from pyro.poutine.util import site_is_subsample


def _sample_posterior(model, first_available_dim, *args, **kwargs):

    with block(), enum(first_available_dim=first_available_dim):
        model_tr = trace(model).get_trace(*args, **kwargs)

    terms = terms_from_trace(model_tr)

    with funsor.interpretations.lazy:
        log_prob = funsor.sum_product.sum_product(
            funsor.ops.max, funsor.ops.add,
            terms["log_measures"] + terms["log_factors"],
            eliminate=terms["measure_vars"] | terms["plate_vars"],
            plates=terms["plate_vars"]
        )
        log_prob = funsor.optimizer.apply_optimizer(log_prob)

    with funsor.approximations.argmax_approximate:
        map_factors = funsor.adjoint.adjoint(funsor.ops.max, funsor.ops.add, log_prob)

    # construct a result trace to replay against the model
    sample_tr = model_tr.copy()
    for name, node in sample_tr.nodes.items():
        if node["type"] != "sample" or node["is_observed"] or site_is_subsample(node):
            continue
        node["funsor"]["log_measure"] = map_factors[node["funsor"]["log_measure"]]
        node["funsor"]["value"] = _get_support_value(node["funsor"]["log_measure"], name)

    with replay(trace=sample_tr):
        return model(*args, **kwargs)


def infer_discrete(model, first_available_dim=None):
    return functools.partial(_sample_posterior, model, first_available_dim)
