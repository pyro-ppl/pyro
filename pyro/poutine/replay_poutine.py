from __future__ import absolute_import, division, print_function

from .poutine import Poutine


class ReplayPoutine(Poutine):
    """
    Poutine for replaying from an existing execution trace.
    """

    def __init__(self, fn, guide_trace, sites=None):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)
        :param guide_trace: a trace whose values should be reused

        Constructor.
        Stores guide_trace in an attribute.
        """
        super(ReplayPoutine, self).__init__(fn)
        assert guide_trace is not None, "must provide guide_trace"
        self.guide_trace = guide_trace
        # case 1: no sites
        if sites is None:
            self.sites = {site: site for site in guide_trace.nodes.keys()
                          if guide_trace.nodes[site]["type"] == "sample" and
                          not guide_trace.nodes[site]["is_observed"]}
        # case 2: sites is a list/tuple/set
        elif isinstance(sites, (list, tuple, set)):
            self.sites = {site: site for site in sites}
        # case 3: sites is a dict
        elif isinstance(sites, dict):
            self.sites = sites
        # otherwise, something is wrong
        # XXX one other possible case: sites is a trace?
        else:
            raise TypeError(
                "unrecognized type {} for sites".format(str(type(sites))))

    def _prepare_site(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the same message, possibly with some fields mutated.

        If the site type is "map_data",
        passes map_data batch indices from the guide trace
        all the way down to the bottom of the stack,
        so that the correct indices are used.

        If the site type is "sample",
        sets the return value and the "done" flag
        so that poutines below it do not execute their sample functions at that site.
        """
        if msg["name"] in self.sites:
            if msg["type"] == "sample" and not msg["is_observed"]:
                msg["done"] = True
                msg["value"] = self.guide_trace.nodes[self.sites[msg["name"]]]["value"]

        return msg

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        At a sample site that appears in self.guide_trace,
        returns the value from self.guide_trace instead of sampling
        from the stochastic function at the site.

        At a sample site that does not appear in self.guide_trace,
        reverts to default Poutine._pyro_sample behavior with no additional side effects.
        """
        name = msg["name"]
        # case 1: dict, positive: sample from guide
        if name in self.sites:
            if msg["is_observed"]:
                raise RuntimeError("site {} is observed and should not be overwritten".format(name))
            g_name = self.sites[name]
            if g_name not in self.guide_trace:
                raise RuntimeError("{} in sites but {} not in trace".format(name, g_name))
            if self.guide_trace.nodes[g_name]["type"] != "sample" or \
                    self.guide_trace.nodes[g_name]["is_observed"]:
                raise RuntimeError("site {} must be sample in guide_trace".format(g_name))
            msg["done"] = True
            return self.guide_trace.nodes[g_name]["value"]
        # case 2: dict, negative: sample from model
        else:
            return super(ReplayPoutine, self)._pyro_sample(msg)
