from __future__ import absolute_import, division, print_function

from .messenger import Messenger


class ReplayMessenger(Messenger):
    """
    Messenger for replaying from an existing execution trace.
    """

    def __init__(self, trace=None, params=None):
        """
        :param trace: a trace whose values should be reused

        Constructor.
        Stores trace in an attribute.
        """
        super(ReplayMessenger, self).__init__()
        if trace is None and params is None:
            raise ValueError("must provide trace or params to replay against")
        self.trace = trace
        self.params = params

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        At a sample site that appears in self.trace,
        returns the value from self.trace instead of sampling
        from the stochastic function at the site.

        At a sample site that does not appear in self.trace,
        reverts to default Messenger._pyro_sample behavior with no additional side effects.
        """
        name = msg["name"]
        if self.trace is not None and name in self.trace:
            guide_msg = self.trace.nodes[name]
            if msg["is_observed"]:
                return None
            if guide_msg["type"] != "sample" or \
                    guide_msg["is_observed"]:
                raise RuntimeError("site {} must be sample in trace".format(name))
            msg["done"] = True
            msg["value"] = guide_msg["value"]
            msg["infer"] = guide_msg["infer"]
        return None

    def _pyro_param(self, msg):
        name = msg["name"]
        if self.params is not None and name in self.params:
            assert hasattr(self.params[name], "unconstrained"), \
                "param {} must be constrained value".format(name)
            msg["done"] = True
            msg["value"] = self.params[name]
        return None
