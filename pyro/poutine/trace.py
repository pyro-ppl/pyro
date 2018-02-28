from __future__ import absolute_import, division, print_function

import collections
import warnings

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


def topological_sort(trace):
    """
    Computes topological ordering of sites in the trace.
    """
    indegree_map = {node: 0 for node in trace.nodes}
    for node_from, node_to in trace.iter_edges():
        indegree_map[node_to] += 1
    zero_indegree = [node for node in indegree_map if indegree_map[node] == 0]
    while len(zero_indegree) > 0:
        node = zero_indegree.pop()
        for child in trace.edges[node]:
            indegree_map[child] -= 1
            if indegree_map[child] == 0:
                zero_indegree.append(child)
                del indegree_map[child]
        yield node


def map_trace(fn, nodes):
    if isinstance(nodes, Trace):
        nodes = nodes.nodes
    ret = collections.OrderedDict()
    for name, node in nodes.items():
        ret[name] = fn(node)
    return ret


def fold_trace(fn, initial_state, nodes):
    if isinstance(nodes, Trace):
        nodes = nodes.nodes
    state = initial_state
    for name, node in nodes.items():
        state = fn(state, node)
    return state


def filter_trace(fn, nodes):
    if isinstance(nodes, Trace):
        nodes = nodes.nodes
    ret = collections.OrderedDict()
    for name, node in nodes.items():
        if fn(node):
            ret[name] = node
    return ret


def log_pdf(site):
    try:
        site_log_p = site["log_pdf"]
    except KeyError:
        args, kwargs = site["args"], site["kwargs"]
        site_log_p = site["fn"].log_prob(site["value"], *args, **kwargs)
        site_log_p = scale_tensor(site_log_p, site["scale"]).sum()
        site["log_pdf"] = site_log_p
        _warn_if_nan(site["name"], site_log_p)
    return site["log_pdf"]


def batch_log_pdf(site):
    try:
        site["batch_log_pdf"]
    except KeyError:
        args, kwargs = site["args"], site["kwargs"]
        site_log_p = site["fn"].log_prob(site["value"], *args, **kwargs)
        site_log_p = scale_tensor(site_log_p, site["scale"])
        site["batch_log_pdf"] = site_log_p
        site["log_pdf"] = site_log_p.sum()
        _warn_if_nan(site["name"], site["log_pdf"])
    return site["batch_log_pdf"]


def score_parts(site):
    try:
        site["score_parts"]
    except KeyError:
        # Note that ScoreParts overloads the multiplication operator
        # to correctly scale each of its three parts.
        value = site["fn"].score_parts(site["value"], *site["args"], **site["kwargs"]) * site["scale"]
        site["score_parts"] = value
        site["batch_log_pdf"] = value[0]
        site["log_pdf"] = value[0].sum()
        _warn_if_nan(site["name"], site["log_pdf"])
    return site["score_parts"]


def is_observed(site):
    return site["type"] == "sample" and \
        site["is_observed"]


def is_sample(site):
    return site["type"] == "sample" and \
        not site["is_observed"]


def is_param(site):
    return site["type"] == "param"


def is_subsample(site):
    return site["type"] == "sample" and \
        type(site["fn"]).__name__ == "_Subsample"


def is_reparameterized(site):
    return site["type"] == "sample" and \
        not site["is_observed"] and \
        getattr(site["fn"], "reparameterized", False)


def is_non_reparameterized(site):
    return site["type"] == "sample" and \
        not site["is_observed"] and \
        not getattr(site["fn"], "reparameterized", False)


class Trace(object):
    """
    Execution trace data structure.
    """

    def __init__(self, graph_type="flat"):
        """
        :param string graph_type: string specifying the kind of trace graph to construct

        Constructor. Stores graph_type attribute and creates data dictionaries.
        """
        assert graph_type in ("flat", "dense"), \
            "{} not a valid graph type".format(graph_type)
        self.graph_type = graph_type
        # metadata
        self.graph = collections.OrderedDict()
        self.nodes = collections.OrderedDict()
        self.edges = collections.OrderedDict()

    def __contains__(self, x):
        return x in self.nodes

    def __getitem__(self, name):
        return self.edges[name]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def successors(self, site_name):
        """
        :param string site_name: name of site to get successors of

        Returns an iterator over site names whose parent is site_name
        """
        return iter(self.edges[site_name])

    def descendants(self, site_name):
        """
        :param string site_name: name of site to get descendants of

        Returns a list of all sites downstream of site_name
        """
        desc = set()
        for succ in self.edges[site_name]:
            desc.add(succ)
            desc = desc.union(set(self.descendants(succ)))
        return list(desc)

    def add_node(self, site_name, **kwargs):
        """
        :param string site_name: the name of the site to be added

        Adds a site with metadata in the kwargs to the trace.
        Raises an error when attempting to add a duplicate sample node.
        """
        # XXX should do more validation than this
        if kwargs["type"] != "param":
            assert site_name not in self, \
                "site {} already in trace".format(site_name)
        self.nodes[site_name] = dict(**kwargs)
        self.edges[site_name] = {}

    def remove_node(self, site_name):
        """
        :param string site_name: the name of the site to be removed

        Deletes node site_name and any edges it belongs to.
        """
        self.nodes.pop(site_name, None)
        self.edges.pop(site_name, None)

    def add_edge(self, node_from, node_to, **kwargs):
        """
        :param string node_from: the name of the parent site
        :param string node_to: the name of the child site

        Adds an edge with optional metadata to the trace.
        Both sites must already be in the trace.
        """
        assert node_from in self, \
            "node_from {} not in trace".format(node_from)
        assert node_to in self, \
            "node_to {} not in trace".format(node_to)
        assert node_to not in self.edges[node_from], \
            "edge from {} to {} already exists".format(node_from, node_to)
        self.edges[node_from][node_to] = dict(**kwargs)

    def remove_edge(self, node_from, node_to):
        """
        :param string node_from: the name of the parent site
        :param string node_to: the name of the child site

        Removes an edge from the trace.
        Both sites must already be in the trace
        """
        assert node_from in self, \
            "node_from {} not in trace".format(node_from)
        assert node_to in self, \
            "node_to {} not in trace".format(node_to)
        assert node_to in self.edges[node_from], \
            "edge from {} to {} must exist".format(node_from, node_to)
        del self.edges[node_from][node_to]

    def copy(self):
        """
        Makes a shallow copy of self with nodes and edges preserved.
        """
        new_trace = Trace(graph_type=self.graph_type)
        new_trace.graph.update(self.graph.copy())
        new_trace.nodes.update(self.nodes.copy())
        new_trace.edges.update(self.edges.copy())
        return new_trace

    def iter_edges(self):
        """
        Iterates (parent, child) pairs corresponding to edges in the trace.
        """
        for node_from, edge_dict in self.edges.items():
            for node_to in edge_dict:
                yield (node_from, node_to)

    def topological_sort(self):
        return topological_sort(self)

    def log_pdf(self, site_filter=lambda site: True):
        """
        Compute the local and overall log-probabilities of the trace.

        The local computation is memoized.

        :returns: total log probability.
        :rtype: torch.autograd.Variable
        """
        return fold_trace(lambda log_p, msg: log_p + log_pdf(msg),
                          0.0, filter_trace(site_filter, self))

    def compute_batch_log_pdf(self, site_filter=lambda site: True):
        """
        Compute the batched local log-probabilities at each site of the trace.

        The local computation is memoized, and also stores the local `.log_pdf()`.
        """
        map_trace(batch_log_pdf, self)

    def compute_score_parts(self):
        """
        Compute the batched local score parts at each site of the trace.
        """
        map_trace(score_parts, self)

    @property
    def observation_nodes(self):
        """
        Gets a list of names of observe sites
        """
        return filter_trace(is_observed, self)

    @property
    def stochastic_nodes(self):
        """
        Gets a list of names of sample sites
        """
        return filter_trace(is_sample, self)

    @property
    def reparameterized_nodes(self):
        """
        Gets a list of names of sample sites whose stochastic functions
        are reparameterizable primitive distributions
        """
        return filter_trace(is_reparameterized, self)

    @property
    def nonreparam_stochastic_nodes(self):
        """
        Gets a list of names of sample sites whose stochastic functions
        are not reparameterizable primitive distributions
        """
        return filter_trace(is_non_reparameterized, self)

    def iter_stochastic_nodes(self):
        """
        Returns an iterator over stochastic nodes in the trace.
        """
        for name, node in filter_trace(is_sample, self):
            yield name, node
