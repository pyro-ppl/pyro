from .poutine import Poutine
from .trace import Trace


class ConditionPoutine(Poutine):
    """
    Adds values at observe sites to condition on data and override sampling
    """
    def __init__(self, fn, data):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)
        :param data: a dict or a Trace

        Constructor. Doesn't do much, just stores the stochastic function
        and the data to condition on.
        """
        self.data = data
        super(ConditionPoutine, self).__init__(fn)

    def _prepare_site(self, msg):
        """
        :param msg: current message at a trace site
        :returns: the updated message at the same trace site

        If we have data at this site, don't sample from the site function
        """
        if msg["name"] in self.data and msg["type"] == "sample":
            msg["done"] = True
        return msg

    def _pyro_observe(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the observed value at the site.

        Implements default pyro.observe Poutine behavior,
        with an additional side effect:
        If msg["name"] is in self.data, raise an error,
        to avoid overwriting existing observations.
        if the observation at the site is not None, return the observation;
        else call the function and return the result.
        """
        name = msg["name"]
        assert name not in self.data, \
            "Should not change values of existing observes..."
        return super(ConditionPoutine, self)._pyro_observe(msg)

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.

        If msg["name"] appears in self.data,
        convert the sample site into an observe site
        whose observed value is the value from self.data[msg["name"]].

        Otherwise, implements default sampling behavior
        with no additional effects.
        """
        name = msg["name"]
        if name in self.data:
            msg["done"] = False
            msg["type"] = "observe"
            if isinstance(self.data, Trace):
                msg["obs"] = self.data.nodes[name]["value"]
            else:
                msg["obs"] = self.data[name]
            return super(ConditionPoutine, self)._pyro_observe(msg)
        else:
            return super(ConditionPoutine, self)._pyro_sample(msg)
