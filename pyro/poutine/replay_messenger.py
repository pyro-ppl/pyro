from .messenger import Messenger


class ReplayMessenger(Messenger):
    """
    Messenger for replaying from a previous execution trace.

    :param trace: An optional trace whose sample sites should be replayed at
        ``pyro.sample`` statements.
    :type trace: ~pyro.poutine.Trace or None
    :param params: An optional dict mapping name to param value to be replayed
        at ``pyro.param`` satements.
    :type params: dict or None
    :param samples: An optional dict mapping name to sample value to be
        replayed at ``pyro.sample`` satements.
    :type params: dict or None
    """

    def __init__(self, trace=None, params=None, samples=None):
        super().__init__()
        if trace is None and params is None and samples is None:
            raise ValueError("must provide trace, params, or samples to replay against")
        self.trace = trace
        self.params = params
        self.samples = samples

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        At a sample site that appears in self.trace,
        returns the value from self.trace instead of sampling
        from the stochastic function at the site.

        At a sample site that does not appear in self.trace,
        reverts to default Messenger._pyro_sample behavior with no additional side effects.
        """
        if msg["is_observed"]:
            return None
        name = msg["name"]
        if self.trace is not None and name in self.trace:
            guide_msg = self.trace.nodes[name]
            if guide_msg["type"] != "sample" or guide_msg["is_observed"]:
                raise RuntimeError("site {} must be sample in trace".format(name))
            msg["done"] = True
            msg["value"] = guide_msg["value"]
            msg["infer"] = guide_msg["infer"]
        elif self.samples is not None and name in self.samples:
            msg["done"] = True
            msg["value"] = self.samples[name]
        return None

    def _pyro_param(self, msg):
        name = msg["name"]
        if self.params is not None and name in self.params:
            assert hasattr(self.params[name], "unconstrained"), \
                "param {} must be constrained value".format(name)
            msg["done"] = True
            msg["value"] = self.params[name]
        return None
