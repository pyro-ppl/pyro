# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
import sys

import opt_einsum

from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.util import scale_and_mask
from pyro.ops.packed import pack
from pyro.poutine.util import is_validation_enabled
from pyro.util import warn_if_inf, warn_if_nan


class Trace:
    """
    Graph data structure denoting the relationships amongst different pyro primitives
    in the execution trace.

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

    Values of ``trace.nodes`` are dictionaries of node metadata:

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

    :param string graph_type: string specifying the kind of trace graph to construct
    """

    def __init__(self, graph_type="flat"):
        assert graph_type in ("flat", "dense"), \
            "{} not a valid graph type".format(graph_type)
        self.graph_type = graph_type
        self.nodes = OrderedDict()
        self._succ = OrderedDict()
        self._pred = OrderedDict()

    def __contains__(self, name):
        return name in self.nodes

    def __iter__(self):
        return iter(self.nodes.keys())

    def __len__(self):
        return len(self.nodes)

    @property
    def edges(self):
        for site, adj_nodes in self._succ.items():
            for adj_node in adj_nodes:
                yield site, adj_node

    def add_node(self, site_name, **kwargs):
        """
        :param string site_name: the name of the site to be added

        Adds a site to the trace.

        Raises an error when attempting to add a duplicate node
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
        self.nodes[site_name] = kwargs
        self._pred[site_name] = set()
        self._succ[site_name] = set()

    def add_edge(self, site1, site2):
        for site in (site1, site2):
            if site not in self.nodes:
                self.add_node(site)
        self._succ[site1].add(site2)
        self._pred[site2].add(site1)

    def remove_node(self, site_name):
        self.nodes.pop(site_name)
        for p in self._pred[site_name]:
            self._succ[p].remove(site_name)
        for s in self._succ[site_name]:
            self._pred[s].remove(site_name)
        self._pred.pop(site_name)
        self._succ.pop(site_name)

    def predecessors(self, site_name):
        return self._pred[site_name]

    def successors(self, site_name):
        return self._succ[site_name]

    def copy(self):
        """
        Makes a shallow copy of self with nodes and edges preserved.
        """
        new_tr = Trace(graph_type=self.graph_type)
        new_tr.nodes.update(self.nodes)
        new_tr._succ.update(self._succ)
        new_tr._pred.update(self._pred)
        return new_tr

    def _dfs(self, site, visited):
        if site in visited:
            return
        for s in self._succ[site]:
            for node in self._dfs(s, visited):
                yield node
        visited.add(site)
        yield site

    def topological_sort(self, reverse=False):
        """
        Return a list of nodes (site names) in topologically sorted order.

        :param bool reverse: Return the list in reverse order.
        :return: list of topologically sorted nodes (site names).
        """
        visited = set()
        top_sorted = []
        for s in self._succ:
            for node in self._dfs(s, visited):
                top_sorted.append(node)
        return top_sorted if reverse else list(reversed(top_sorted))

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
                if "log_prob_sum" in site:
                    log_p = site["log_prob_sum"]
                else:
                    try:
                        log_p = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
                    except ValueError as e:
                        _, exc_value, traceback = sys.exc_info()
                        shapes = self.format_shapes(last_site=site["name"])
                        raise ValueError("Error while computing log_prob_sum at site '{}':\n{}\n{}\n"
                                         .format(name, exc_value, shapes)).with_traceback(traceback) from e
                    log_p = scale_and_mask(log_p, site["scale"], site["mask"]).sum()
                    site["log_prob_sum"] = log_p
                    if is_validation_enabled():
                        warn_if_nan(log_p, "log_prob_sum at site '{}'".format(name))
                        warn_if_inf(log_p, "log_prob_sum at site '{}'".format(name), allow_neginf=True)
                result = result + log_p
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
                if "log_prob" not in site:
                    try:
                        log_p = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
                    except ValueError as e:
                        _, exc_value, traceback = sys.exc_info()
                        shapes = self.format_shapes(last_site=site["name"])
                        raise ValueError("Error while computing log_prob at site '{}':\n{}\n{}"
                                         .format(name, exc_value, shapes)).with_traceback(traceback) from e
                    site["unscaled_log_prob"] = log_p
                    log_p = scale_and_mask(log_p, site["scale"], site["mask"])
                    site["log_prob"] = log_p
                    site["log_prob_sum"] = log_p.sum()
                    if is_validation_enabled():
                        warn_if_nan(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name))
                        warn_if_inf(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name),
                                    allow_neginf=True)

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
                try:
                    value = site["fn"].score_parts(site["value"], *site["args"], **site["kwargs"])
                except ValueError as e:
                    _, exc_value, traceback = sys.exc_info()
                    shapes = self.format_shapes(last_site=site["name"])
                    raise ValueError("Error while computing score_parts at site '{}':\n{}\n{}"
                                     .format(name, exc_value, shapes)).with_traceback(traceback) from e
                site["unscaled_log_prob"] = value.log_prob
                value = value.scale_and_mask(site["scale"], site["mask"])
                site["score_parts"] = value
                site["log_prob"] = value.log_prob
                site["log_prob_sum"] = value.log_prob.sum()
                if is_validation_enabled():
                    warn_if_nan(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name))
                    warn_if_inf(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name), allow_neginf=True)

    def detach_(self):
        """
        Detach values (in-place) at each sample site of the trace.
        """
        for _, site in self.nodes.items():
            if site["type"] == "sample":
                site["value"] = site["value"].detach()

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

    def symbolize_dims(self, plate_to_symbol=None):
        """
        Assign unique symbols to all tensor dimensions.
        """
        plate_to_symbol = {} if plate_to_symbol is None else plate_to_symbol
        symbol_to_dim = {}
        for site in self.nodes.values():
            if site["type"] != "sample":
                continue

            # allocate even symbols for plate dims
            dim_to_symbol = {}
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    if frame.name in plate_to_symbol:
                        symbol = plate_to_symbol[frame.name]
                    else:
                        symbol = opt_einsum.get_symbol(2 * len(plate_to_symbol))
                        plate_to_symbol[frame.name] = symbol
                    symbol_to_dim[symbol] = frame.dim
                    dim_to_symbol[frame.dim] = symbol

            # allocate odd symbols for enum dims
            for dim, id_ in site["infer"].get("_dim_to_id", {}).items():
                symbol = opt_einsum.get_symbol(1 + 2 * id_)
                symbol_to_dim[symbol] = dim
                dim_to_symbol[dim] = symbol
            enum_dim = site["infer"].get("_enumerate_dim")
            if enum_dim is not None:
                site["infer"]["_enumerate_symbol"] = dim_to_symbol[enum_dim]
            site["infer"]["_dim_to_symbol"] = dim_to_symbol

        self.plate_to_symbol = plate_to_symbol
        self.symbol_to_dim = symbol_to_dim

    def pack_tensors(self, plate_to_symbol=None):
        """
        Computes packed representations of tensors in the trace.
        This should be called after :meth:`compute_log_prob` or :meth:`compute_score_parts`.
        """
        self.symbolize_dims(plate_to_symbol)
        for site in self.nodes.values():
            if site["type"] != "sample":
                continue
            dim_to_symbol = site["infer"]["_dim_to_symbol"]
            packed = site.setdefault("packed", {})
            try:
                packed["mask"] = pack(site["mask"], dim_to_symbol)
                if "score_parts" in site:
                    log_prob, score_function, entropy_term = site["score_parts"]
                    log_prob = pack(log_prob, dim_to_symbol)
                    score_function = pack(score_function, dim_to_symbol)
                    entropy_term = pack(entropy_term, dim_to_symbol)
                    packed["score_parts"] = ScoreParts(log_prob, score_function, entropy_term)
                    packed["log_prob"] = log_prob
                    packed["unscaled_log_prob"] = pack(site["unscaled_log_prob"], dim_to_symbol)
                elif "log_prob" in site:
                    packed["log_prob"] = pack(site["log_prob"], dim_to_symbol)
                    packed["unscaled_log_prob"] = pack(site["unscaled_log_prob"], dim_to_symbol)
            except ValueError as e:
                _, exc_value, traceback = sys.exc_info()
                shapes = self.format_shapes(last_site=site["name"])
                raise ValueError("Error while packing tensors at site '{}':\n  {}\n{}"
                                 .format(site["name"], exc_value, shapes)).with_traceback(traceback) from e

    def format_shapes(self, title='Trace Shapes:', last_site=None):
        """
        Returns a string showing a table of the shapes of all sites in the
        trace.
        """
        if not self.nodes:
            return title
        rows = [[title]]

        rows.append(['Param Sites:'])
        for name, site in self.nodes.items():
            if site["type"] == "param":
                rows.append([name, None] + [str(size) for size in site["value"].shape])
            if name == last_site:
                break

        rows.append(['Sample Sites:'])
        for name, site in self.nodes.items():
            if site["type"] == "sample":
                # param shape
                batch_shape = getattr(site["fn"], "batch_shape", ())
                event_shape = getattr(site["fn"], "event_shape", ())
                rows.append([name + " dist", None] + [str(size) for size in batch_shape] +
                            ["|", None] + [str(size) for size in event_shape])

                # value shape
                event_dim = len(event_shape)
                shape = getattr(site["value"], "shape", ())
                batch_shape = shape[:len(shape) - event_dim]
                event_shape = shape[len(shape) - event_dim:]
                rows.append(["value", None] + [str(size) for size in batch_shape] +
                            ["|", None] + [str(size) for size in event_shape])

                # log_prob shape
                if "log_prob" in site:
                    batch_shape = getattr(site["log_prob"], "shape", ())
                    rows.append(["log_prob", None] + [str(size) for size in batch_shape] + ["|", None])
            if name == last_site:
                break

        return _format_table(rows)


def _format_table(rows):
    """
    Formats a right justified table using None as column separator.
    """
    # compute column widths
    column_widths = [0, 0, 0]
    for row in rows:
        widths = [0, 0, 0]
        j = 0
        for cell in row:
            if cell is None:
                j += 1
            else:
                widths[j] += 1
        for j in range(3):
            column_widths[j] = max(column_widths[j], widths[j])

    # justify columns
    for i, row in enumerate(rows):
        cols = [[], [], []]
        j = 0
        for cell in row:
            if cell is None:
                j += 1
            else:
                cols[j].append(cell)
        cols = [[""] * (width - len(col)) + col
                if direction == 'r' else
                col + [""] * (width - len(col))
                for width, col, direction in zip(column_widths, cols, 'rrl')]
        rows[i] = sum(cols, [])

    # compute cell widths
    cell_widths = [0] * len(rows[0])
    for row in rows:
        for j, cell in enumerate(row):
            cell_widths[j] = max(cell_widths[j], len(cell))

    # justify cells
    return "\n".join(" ".join(cell.rjust(width)
                              for cell, width in zip(row, cell_widths))
                     for row in rows)
