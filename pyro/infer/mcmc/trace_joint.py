from abc import ABCMeta, abstractmethod

from six import add_metaclass

from pyro.distributions.empirical import Empirical


@add_metaclass(ABCMeta)
class TraceJoint(object):
    """
    TODO: merge with TracePosterior
    """
    def __init__(self):
        self._reset()

    def _reset(self):
        self.traces = []
        self.marginal_dist = {}

    @abstractmethod
    def _traces(self, *args, **kwargs):
        return NotImplementedError

    def run(self, *args, **kwargs):
        self._reset()
        for tr in self._traces(*args, **kwargs):
            self.traces.append(tr)
        return self

    def marginal(self, site="_RETURN"):
        """
        Return the marginal for a sample site from the collection of execution traces.
        This is cached for any future calls.

        :param site: sample site whose marginal to return.
        :return: ``dist.Empirical``
        """
        if site in self.marginal_dist:
            return self.marginal_dist[site]
        marginal_site = Empirical()
        for tr in self.traces:
            marginal_site.add(tr.nodes[site]["value"])
        self.marginal_dist[site] = marginal_site
        return marginal_site
