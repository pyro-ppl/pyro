from __future__ import absolute_import, division, print_function

import collections
import warnings

import networkx
from torch.autograd import Variable

from pyro.distributions.util import scale_tensor
from pyro.util import is_nan, is_inf


def _warn_if_nan(name, value):
    if isinstance(value, Variable):
        value = value.data[0]
    if is_nan(value):
        warnings.warn("Encountered NAN log_pdf at site '{}'".format(name))
    if is_inf(value) and value > 0:
        warnings.warn("Encountered +inf log_pdf at site '{}'".format(name))
    # Note that -inf log_pdf is fine: it is merely a zero-probability event.


class DiGraph(networkx.DiGraph):
    node_dict_factory = collections.OrderedDict


class Trace(object):
    """
    Execution trace data structure
    """

    def __init__(self, *args, **kwargs):
        """
        :param string graph_type: string specifying the kind of trace graph to construct

        Constructor. Currently identical to networkx.``DiGraph(\*args, \**kwargs)``,
        except for storing the graph_type attribute
        """
        self._graph = DiGraph(*args, **kwargs)
        graph_type = kwargs.pop("graph_type", "flat")
        assert graph_type in ("flat", "dense"), \
            "{} not a valid graph type".format(graph_type)
        self.graph_type = graph_type
        super(Trace, self).__init__(*args, **kwargs)

    def __del__(self):
        # Work around cyclic reference bugs in networkx.DiGraph
        # See https://github.com/uber/pyro/issues/798
        self._graph.__dict__.clear()

    @property
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        return self._graph.edges

    @property
    def graph(self):
        return self._graph.graph

    @property
    def remove_node(self):
        return self._graph.remove_node

    @property
    def add_edge(self):
        return self._graph.add_edge

    @property
    def is_directed(self):
        return self._graph.is_directed

    @property
    def in_degree(self):
        return self._graph.in_degree

    @property
    def successors(self):
        return self._graph.successors

    def __contains__(self, site_name):
        return site_name in self._graph

    def __iter__(self):
        return iter(self._graph)

    def __len__(self):
        return len(self._graph)

    def add_node(self, site_name, *args, **kwargs):
        """
        :param string site_name: the name of the site to be added

        Adds a site to the trace.

        Identical to super(Trace, self).add_node,
        but raises an error when attempting to add a duplicate node
        instead of silently overwriting.
        """
        # XXX should do more validation than this
        if kwargs["type"] != "param":
            assert site_name not in self, \
                "site {} already in trace".format(site_name)

        # XXX should copy in case site gets mutated, or dont bother?
        self._graph.add_node(site_name, *args, **kwargs)

    def copy(self):
        """
        Makes a shallow copy of self with nodes and edges preserved.
        Identical to super(Trace, self).copy(), but preserves the type
        and the self.graph_type attribute
        """
        trace = Trace()
        trace._graph = self._graph.copy()
        trace.graph_type = self.graph_type
        return trace

    def log_pdf(self, site_filter=lambda name, site: True):
        """
        Compute the local and overall log-probabilities of the trace.

        The local computation is memoized.

        :returns: total log probability.
        :rtype: torch.autograd.Variable
        """
        log_p = 0.0
        for name, site in self.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                try:
                    site_log_p = site["log_pdf"]
                except KeyError:
                    args, kwargs = site["args"], site["kwargs"]
                    site_log_p = site["fn"].log_prob(site["value"], *args, **kwargs)
                    site_log_p = scale_tensor(site_log_p, site["scale"]).sum()
                    site["log_pdf"] = site_log_p
                    _warn_if_nan(name, site_log_p)
                log_p += site_log_p
        return log_p

    # XXX This only makes sense when all tensors have compatible shape.
    def batch_log_pdf(self, site_filter=lambda name, site: True):
        """
        Compute the batched local and overall log-probabilities of the trace.

        The local computation is memoized, and also stores the local `.log_pdf()`.
        """
        log_p = 0.0
        for name, site in self.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                try:
                    site_log_p = site["batch_log_pdf"]
                except KeyError:
                    args, kwargs = site["args"], site["kwargs"]
                    site_log_p = site["fn"].log_prob(site["value"], *args, **kwargs)
                    site_log_p = scale_tensor(site_log_p, site["scale"])
                    site["batch_log_pdf"] = site_log_p
                    site["log_pdf"] = site_log_p.sum()
                    _warn_if_nan(name, site["log_pdf"])
                # Here log_p may be broadcast to a larger tensor:
                log_p = log_p + site_log_p
        return log_p

    def compute_batch_log_pdf(self, site_filter=lambda name, site: True):
        """
        Compute the batched local log-probabilities at each site of the trace.

        The local computation is memoized, and also stores the local `.log_pdf()`.
        """
        for name, site in self.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                try:
                    site["batch_log_pdf"]
                except KeyError:
                    args, kwargs = site["args"], site["kwargs"]
                    site_log_p = site["fn"].log_prob(site["value"], *args, **kwargs)
                    site_log_p = scale_tensor(site_log_p, site["scale"])
                    site["batch_log_pdf"] = site_log_p
                    site["log_pdf"] = site_log_p.sum()
                    _warn_if_nan(name, site["log_pdf"])

    def compute_score_parts(self):
        """
        Compute the batched local score parts at each site of the trace.
        """
        for name, site in self.nodes.items():
            if site["type"] == "sample" and "score_parts" not in site:
                # Note that ScoreParts overloads the multiplication operator
                # to correctly scale each of its three parts.
                value = site["fn"].score_parts(site["value"], *site["args"], **site["kwargs"]) * site["scale"]
                site["score_parts"] = value
                site["batch_log_pdf"] = value[0]
                site["log_pdf"] = value[0].sum()
                _warn_if_nan(name, site["log_pdf"])

    @property
    def observation_nodes(self):
        """
        Gets a list of names of observe sites
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                node["is_observed"]]

    @property
    def stochastic_nodes(self):
        """
        Gets a list of names of sample sites
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                not node["is_observed"]]

    @property
    def reparameterized_nodes(self):
        """
        Gets a list of names of sample sites whose stochastic functions
        are reparameterizable primitive distributions
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                not node["is_observed"] and
                getattr(node["fn"], "reparameterized", False)]

    @property
    def nonreparam_stochastic_nodes(self):
        """
        Gets a list of names of sample sites whose stochastic functions
        are not reparameterizable primitive distributions
        """
        return list(set(self.stochastic_nodes) - set(self.reparameterized_nodes))

    def iter_stochastic_nodes(self):
        """
        Returns an iterator over stochastic nodes in the trace.
        """
        for name, node in self.nodes.items():
            if node["type"] == "sample" and not node["is_observed"]:
                yield name, node
