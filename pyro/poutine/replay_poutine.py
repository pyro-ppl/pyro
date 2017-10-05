import pyro

from .poutine import Poutine
from .lambda_poutine import LambdaPoutine


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
            self.sites = {site: site for site in guide_trace.keys()}
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
            if msg["type"] == "map_data":
                guide_type = self.guide_trace[msg["name"]]["type"]
                assert self.guide_trace[msg["name"]]["type"] == "map_data", \
                    "{} is {}, not map_data, in guide_trace".format(msg["name"],
                                                                    guide_type)
                msg["indices"] = self.guide_trace[msg["name"]]["indices"]
                msg["batch_size"] = self.guide_trace[msg["name"]]["batch_size"]
                msg["batch_dim"] = self.guide_trace[msg["name"]]["batch_dim"]

            # dont reexecute
            if msg["type"] == "sample":
                msg["done"] = True
                msg["ret"] = self.guide_trace[msg["name"]]["value"]

        return msg

    def _pyro_sample(self, msg):  # , name, fn, *args, **kwargs):
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
            g_name = self.sites[name]
            assert g_name in self.guide_trace, \
                "{} in sites but {} not in trace".format(name, g_name)
            assert self.guide_trace[g_name]["type"] == "sample", \
                "site {} must be sample in guide_trace".format(g_name)
            # msg["done"] = True
            return self.guide_trace[g_name]["value"]
        # case 2: dict, negative: sample from model
        elif name not in self.sites:
            return super(ReplayPoutine, self)._pyro_sample(msg)  # , name, fn, *args, **kwargs)
        else:
            raise ValueError(
                "something went wrong with replay conditions at site " + name)

    def _pyro_map_data(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the result of running the site function on the data.

        Instead of sampling new batch indices,
        uses the batch indices from the guide trace,
        already provided by _prepare_site.
        """
        name, data, fn, batch_size, batch_dim = \
            msg["name"], msg["data"], msg["fn"], msg["batch_size"], msg["batch_dim"]
        scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        msg["fn"] = LambdaPoutine(fn, name, scale)
        ret = super(ReplayPoutine, self)._pyro_map_data(msg)
        msg["fn"] = fn
        return ret
