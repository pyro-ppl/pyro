from __future__ import absolute_import, division, print_function

import collections

import networkx

from pyro.distributions.util import scale_and_mask
from pyro.poutine.util import is_validation_enabled
from pyro.util import warn_if_nan, warn_if_inf


class Trace(networkx.DiGraph):
    """
    Execution trace data structure built on top of :class:`networkx.DiGraph`.

    An execution trace of a Pyro program is a record of every call
    to ``pyro.sample()`` and ``pyro.param()`` in a single execution of that program.
    Traces are directed graphs whose nodes represent primitive calls or input/output,
    and whose edges represent conditional dependence relationships
    between those primitive calls. They are created and populated by ``poutine.trace``.

    Each node (or site) in a trace contains the name, input and output value of the site,
    as well as additional metadata added by inference algorithms or user annotation.
    In the case of ``pyro.sample``, the trace also includes the stochastic function
    at the site, and any observed data added by users.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    We can record its execution using ``pyro.poutine.trace``
    and use the resulting data structure to compute the log-joint probability
    of all of the sample sites in the execution or extract all parameters.

        >>> trace = pyro.poutine.trace(model).get_trace(0.0)
        >>> logp = trace.log_prob_sum()
        >>> params = [trace.nodes[name]["value"].unconstrained() for name in trace.param_nodes]

    We can also inspect or manipulate individual nodes in the trace.
    ``trace.nodes`` contains a ``collections.OrderedDict``
    of site names and metadata corresponding to ``x``, ``s``, ``z``, and the return value:

        >>> list(name for name in trace.nodes.keys())  # doctest: +SKIP
        ["_INPUT", "s", "z", "_RETURN"]

    As in :class:`networkx.DiGraph`, values of ``trace.nodes`` are dictionaries of node metadata:

        >>> trace.nodes["z"]  # doctest: +SKIP
        {'type': 'sample', 'name': 'z', 'is_observed': False,
         'fn': Normal(), 'value': tensor(0.6480), 'args': (), 'kwargs': {},
         'infer': {}, 'scale': 1.0, 'cond_indep_stack': (),
         'done': True, 'stop': False, 'continuation': None}

    ``'infer'`` is a dictionary of user- or algorithm-specified metadata.
    ``'args'`` and ``'kwargs'`` are the arguments passed via ``pyro.sample``
    to ``fn.__call__`` or ``fn.log_prob``.
    ``'scale'`` is used to scale the log-probability of the site when computing the log-joint.
    ``'cond_indep_stack'`` contains data structures corresponding to ``pyro.plate`` contexts
    appearing in the execution.
    ``'done'``, ``'stop'``, and ``'continuation'`` are only used by Pyro's internals.
    """

    node_dict_factory = collections.OrderedDict

    def __init__(self, *args, **kwargs):
        """
        :param string graph_type: string specifying the kind of trace graph to construct

        Constructor. Currently identical to :meth:`networkx.DiGraph.__init__`,
        except for storing the graph_type attribute
        """
        graph_type = kwargs.pop("graph_type", "flat")
        assert graph_type in ("flat", "dense"), \
            "{} not a valid graph type".format(graph_type)
        self.graph_type = graph_type
        super(Trace, self).__init__(*args, **kwargs)

    def add_node(self, site_name, *args, **kwargs):
        """
        :param string site_name: the name of the site to be added

        Adds a site to the trace.

        Identical to :meth:`networkx.DiGraph.add_node`
        but raises an error when attempting to add a duplicate node
        instead of silently overwriting.
        """
        if site_name in self:
            site = self.nodes[site_name]
            if site['type'] != kwargs['type']:
                # Cannot sample or observe after a param statement.
                raise RuntimeError("{} is already in the trace as a {}".format(site_name, site['type']))
            elif kwargs['type'] != "param":
                # Cannot sample after a previous sample statement.
                raise RuntimeError("Multiple {} sites named '{}'".format(kwargs['type'], site_name))

        # XXX should copy in case site gets mutated, or dont bother?
        super(Trace, self).add_node(site_name, *args, **kwargs)

    def copy(self):
        """
        Makes a shallow copy of self with nodes and edges preserved.
        Identical to :meth:`networkx.DiGraph.copy`, but preserves the type
        and the self.graph_type attribute
        """
        trace = super(Trace, self).copy()
        trace.__class__ = Trace
        trace.graph_type = self.graph_type
        return trace

    def log_prob_sum(self, site_filter=lambda name, site: True):
        """
        Compute the site-wise log probabilities of the trace.
        Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
        Each ``log_prob_sum`` is a scalar.
        The computation of ``log_prob_sum`` is memoized.

        :returns: total log probability.
        :rtype: torch.Tensor
        """
        result = 0.0
        for name, site in self.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                try:
                    log_p = site["log_prob_sum"]
                except KeyError:
                    log_p = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
                    log_p = scale_and_mask(log_p, site["scale"], site["mask"]).sum()
                    site["log_prob_sum"] = log_p
                    if is_validation_enabled():
                        warn_if_nan(log_p, "log_prob_sum at site '{}'".format(name))
                        warn_if_inf(log_p, "log_prob_sum at site '{}'".format(name), allow_neginf=True)
                result += log_p
        return result

    def compute_log_prob(self, site_filter=lambda name, site: True):
        """
        Compute the site-wise log probabilities of the trace.
        Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
        Each ``log_prob_sum`` is a scalar.
        Both computations are memoized.
        """
        for name, site in self.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                try:
                    site["log_prob"]
                except KeyError:
                    log_p = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
                    site["unscaled_log_prob"] = log_p
                    log_p = scale_and_mask(log_p, site["scale"], site["mask"])
                    site["log_prob"] = log_p
                    site["log_prob_sum"] = log_p.sum()
                    if is_validation_enabled():
                        warn_if_nan(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name))
                        warn_if_inf(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name), allow_neginf=True)

    def compute_score_parts(self):
        """
        Compute the batched local score parts at each site of the trace.
        Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
        Each ``log_prob_sum`` is a scalar.
        All computations are memoized.
        """
        for name, site in self.nodes.items():
            if site["type"] == "sample" and "score_parts" not in site:
                # Note that ScoreParts overloads the multiplication operator
                # to correctly scale each of its three parts.
                value = site["fn"].score_parts(site["value"], *site["args"], **site["kwargs"])
                site["unscaled_log_prob"] = value.log_prob
                value = value.scale_and_mask(site["scale"], site["mask"])
                site["score_parts"] = value
                site["log_prob"] = value.log_prob
                site["log_prob_sum"] = value.log_prob.sum()
                if is_validation_enabled():
                    warn_if_nan(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name))
                    warn_if_inf(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name), allow_neginf=True)

    @property
    def observation_nodes(self):
        """
        :return: a list of names of observe sites
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                node["is_observed"]]

    @property
    def param_nodes(self):
        """
        :return: a list of names of param sites
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "param"]

    @property
    def stochastic_nodes(self):
        """
        :return: a list of names of sample sites
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                not node["is_observed"]]

    @property
    def reparameterized_nodes(self):
        """
        :return: a list of names of sample sites whose stochastic functions
            are reparameterizable primitive distributions
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                not node["is_observed"] and
                getattr(node["fn"], "has_rsample", False)]

    @property
    def nonreparam_stochastic_nodes(self):
        """
        :return: a list of names of sample sites whose stochastic functions
            are not reparameterizable primitive distributions
        """
        return list(set(self.stochastic_nodes) - set(self.reparameterized_nodes))

    def iter_stochastic_nodes(self):
        """
        :return: an iterator over stochastic nodes in the trace.
        """
        for name, node in self.nodes.items():
            if node["type"] == "sample" and not node["is_observed"]:
                yield name, node
