# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
from pyro.poutine.util import site_is_subsample

from .runtime import _PYRO_STACK
from .trace_messenger import TraceMessenger

# TODO Remove import guard once funsor is a required dependency.
try:
    import funsor
    from funsor.terms import Funsor
except ImportError:
    pass


class CollapseMessenger(TraceMessenger):
    def __init__(self, *args, **kwargs):
        import funsor
        funsor.set_backend("torch")
        super().__init__(*args, **kwargs)

    def _process_message(self, msg):
        if site_is_subsample(msg):
            return
        super()._process_message(msg)

        # Block sample statements.
        if msg["type"] == "sample":
            if isinstance(msg["fn"], Funsor) or isinstance(msg["value"], (str, Funsor)):
                msg["stop"] = True

    def _pyro_sample(self, msg):
        if msg["value"] is None:
            msg["value"] = msg["name"]
        msg["done"] = True

    def _pyro_post_sample(self, msg):
        if site_is_subsample(msg):
            return
        super()._pyro_post_sample(msg)

    def __enter__(self):
        self.preserved_plates = frozenset(h.name for h in _PYRO_STACK
                                          if isinstance(h, pyro.plate))
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)

        # Convert delayed statements to pyro.factor()
        reduced_vars = []
        log_prob_terms = []
        plates = frozenset()
        for name, site in self.trace.nodes.items():
            if not site["is_observed"]:
                reduced_vars.append(name)
            log_prob_terms.append(funsor.to_funsor(site["fn"])(value=site["value"]))
            plates |= frozenset(f.name for f in site["cond_indep_stack"]
                                if f.vectorized)
        assert log_prob_terms, "nothing to collapse"
        reduced_plates = plates - self.preserved_plates
        log_prob = funsor.sum_product.sum_product(
            funsor.ops.logaddexp,
            funsor.ops.add,
            log_prob_terms,
            eliminate=frozenset(reduced_vars) | reduced_plates,
            plates=plates,
        )
        name = reduced_vars[0]
        pyro.factor(name, log_prob.data)
