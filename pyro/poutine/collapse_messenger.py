import pyro.distributions as dist

from .runtime import _PYRO_STACK

# TODO Remove import guard once funsor is a required dependency.
try:
    import funsor
    from funsor.terms import Funsor
except ImportError:
    pass


class CollapseMessenger(TraceMessenger):

    def _process_message(self, msg):
        super()._process_message(msg)
        if msg["type"] == "sample":
            if isinstance(msg["fn"], Funsor) or isinstance(msg["value"], Funsor):
                msg["stop"] = True

    def _pyro_sample(self, msg):
        if msg["value"] is not None:
            msg["value"] = funsor.Variable(msg["name"], msg["fn"].value_domain)
        msg["done"] = True

    def __enter__(self):
        self.preserved_plates = frozenset(h.name for h in _PYRO_STACK
                                          if isinstance(h, pyro.plate))
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)

        # Convert delayed statements to pyro.factor()
        log_prob_terms, reduced_vars = [], []
        for name, site in self.trace.items():
            if not site["is_observed"]:
                reduced_vars.append(name)
            log_prob = to_funsor(site["fn"])(value=site["value"])
            plates |= frozenset(f.name for f in site["cond_indep_stack"]
                                if f.vectorized)
        if log_prob_terms:
            reduced_plates = plates - self.preserved_plates
            log_prob = funsor.sum_product.sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                log_prob_terms,
                eliminate=frozenset(reduced_vars) | reduced_plates,
                plates=plates,
            )
            name = reduced_vars[0]
            pyro.factor(name, log_prob)
