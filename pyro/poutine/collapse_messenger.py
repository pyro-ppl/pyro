# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce, singledispatch

import pyro
from pyro.distributions.distribution import COERCIONS
from pyro.poutine.util import site_is_subsample

from .runtime import _PYRO_STACK
from .trace_messenger import TraceMessenger

# TODO Remove import guard once funsor is a required dependency.
try:
    import funsor
    from funsor.cnf import Contraction
    from funsor.delta import Delta
    from funsor.terms import Funsor, Variable
except ImportError:
    # Create fake types for singledispatch.
    Contraction = type("Contraction", (), {})
    Delta = type("Delta", (), {})
    Funsor = type("Funsor", (), {})
    Variable = type("Variable", (), {})


@singledispatch
def _get_free_vars(x):
    return x


@_get_free_vars.register(Variable)
def _(x):
    return frozenset((x.name,))


@_get_free_vars.register(tuple)
def _(x, subs):
    return frozenset().union(*map(_get_free_vars, x))


@singledispatch
def _substitute(x, subs):
    return x


@_substitute.register(str)
def _(x, subs):
    return subs.get(x, x)


@_substitute.register(Variable)
def _(x, subs):
    return subs.get(x.name, x)


@_substitute.register(tuple)
def _(x, subs):
    return tuple(_substitute(part, subs) for part in x)


@singledispatch
def _extract_deltas(f):
    raise NotImplementedError("unmatched {}".format(type(f).__name__))


@_extract_deltas.register(Delta)
def _(f):
    return f


@_extract_deltas.register(Contraction)
def _(f):
    for d in f.terms:
        if isinstance(d, Delta):
            return d


class CollapseMessenger(TraceMessenger):
    """
    EXPERIMENTAL Collapses all sites in the context by lazily sampling and
    attempting to use conjugacy relations. If no conjugacy is known this will
    fail. Code using the results of sample sites must be written to accept
    Funsors rather than Tensors. This requires ``funsor`` to be installed.

    .. warning:: This is not compatible with automatic guessing of
        ``max_plate_nesting``. If any plates appear within the collapsed
        context, you should manually declare ``max_plate_nesting`` to your
        inference algorithm (e.g. ``Trace_ELBO(max_plate_nesting=1)``).
    """
    _coerce = None

    def __init__(self, *args, **kwargs):
        if CollapseMessenger._coerce is None:
            import funsor
            from funsor.distribution import CoerceDistributionToFunsor
            funsor.set_backend("torch")
            CollapseMessenger._coerce = CoerceDistributionToFunsor("torch")
        self._block = False
        super().__init__(*args, **kwargs)

    def _process_message(self, msg):
        if self._block:
            return
        if site_is_subsample(msg):
            return
        super()._process_message(msg)

    def _pyro_sample(self, msg):
        # Eagerly convert fn and value to Funsor.
        dim_to_name = {f.dim: f.name for f in msg["cond_indep_stack"]}
        dim_to_name.update(self.preserved_plates)
        msg["fn"] = funsor.to_funsor(msg["fn"], funsor.Real, dim_to_name)
        domain = msg["fn"].inputs["value"]
        if msg["value"] is None:
            msg["value"] = funsor.Variable(msg["name"], domain)
        else:
            msg["value"] = funsor.to_funsor(msg["value"], domain, dim_to_name)

        msg["done"] = True
        msg["stop"] = True

    def _pyro_post_sample(self, msg):
        if self._block:
            return
        if site_is_subsample(msg):
            return
        super()._pyro_post_sample(msg)

    def _pyro_barrier(self, msg):
        # Get log_prob and record factor.
        name, log_prob, log_joint, sampled_vars = self._get_log_prob()
        self._block = True
        pyro.factor(name, log_prob.data)
        self._block = False

        # Sample
        if sampled_vars:
            samples = log_joint.sample(sampled_vars)
            deltas = _extract_deltas(samples)
            samples = {name: point.data for name, (point, _) in deltas.terms}
        else:
            samples = {}

        # Update value.
        assert len(msg["args"]) == 1
        value = msg["args"][0]
        value = _substitute(value, samples)
        msg["value"] = value

    def __enter__(self):
        self.preserved_plates = {h.dim: h.name for h in _PYRO_STACK
                                 if isinstance(h, pyro.plate)}
        COERCIONS.append(self._coerce)
        return super().__enter__()

    def __exit__(self, *args):
        _coerce = COERCIONS.pop()
        assert _coerce is self._coerce
        super().__exit__(*args)

        if any(site["type"] == "sample"
               for site in self.trace.nodes.values()):
            name, log_prob, _, _ = self._get_log_prob()
            pyro.factor(name, log_prob.data)

    def _get_log_prob(self):
        # Convert delayed statements to pyro.factor()
        reduced_vars = []
        log_prob_terms = []
        plates = frozenset()
        for name, site in self.trace.nodes.items():
            if not site["is_observed"]:
                reduced_vars.append(name)
            log_prob_terms.append(site["fn"](value=site["value"]))
            plates |= frozenset(f.name for f in site["cond_indep_stack"]
                                if f.vectorized)
        name = reduced_vars[0]
        reduced_vars = frozenset(reduced_vars)
        assert log_prob_terms, "nothing to collapse"
        self.trace.nodes.clear()
        reduced_plates = plates - frozenset(self.preserved_plates.values())
        if reduced_plates:
            log_prob = funsor.sum_product.sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                log_prob_terms,
                eliminate=reduced_vars | reduced_plates,
                plates=plates,
            )
            log_joint = NotImplemented
        else:
            log_joint = reduce(funsor.ops.add, log_prob_terms)
            log_prob = log_joint.reduce(funsor.ops.logaddexp, reduced_vars)

        return name, log_prob, log_joint, reduced_vars
