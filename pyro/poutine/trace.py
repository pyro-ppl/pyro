import graphviz
import networkx


class Trace(dict):
    """
    Execution trace data structure
    """

    def add_sample(self, name, scale, val, fn, *args, **kwargs):
        """
        Sample site
        """
        assert name not in self, "sample {} already in trace".format(name)
        site = {}
        site["type"] = "sample"
        site["value"] = val
        site["fn"] = fn
        site["args"] = (args, kwargs)
        site["scale"] = scale
        self[name] = site
        return self

    def add_observe(self, name, scale, val, fn, obs, *args, **kwargs):
        """
        Observe site
        """
        assert name not in self, "observe {} already in trace".format(name)
        site = {}
        site["type"] = "observe"
        site["value"] = val
        site["fn"] = fn
        site["obs"] = obs
        site["args"] = (args, kwargs)
        site["scale"] = scale
        self[name] = site
        return self

    def add_map_data(self, name, fn, batch_size, batch_dim, ind):
        """
        map_data site
        """
        assert name not in self, "map_data {} already in trace".format(name)
        site = {}
        site["type"] = "map_data"
        site["indices"] = ind
        site["batch_size"] = batch_size
        site["batch_dim"] = batch_dim
        site["fn"] = fn
        self[name] = site
        return self

    def add_param(self, name, val, *args, **kwargs):
        """
        param site
        """
        site = {}
        site["type"] = "param"
        site["value"] = val
        site["args"] = (args, kwargs)
        self[name] = site
        return self

    def add_args(self, args_and_kwargs):
        """
        input arguments site
        """
        name = "_INPUT"
        assert name not in self, "_INPUT already in trace"
        site = {}
        site["type"] = "args"
        site["args"] = args_and_kwargs
        self[name] = site
        return self

    def add_return(self, val, *args, **kwargs):
        """
        return value site
        """
        name = "_RETURN"
        assert name not in self, "_RETURN already in trace"
        site = {}
        site["type"] = "return"
        site["value"] = val
        self[name] = site
        return self

    def copy(self):
        """
        Make a copy (for dynamic programming)
        """
        return Trace(self)

    def log_pdf(self, vec_batch_nodes_dict={}):
        """
        Compute the local and overall log-probabilities of the trace
        """

        log_p = 0.0
        for name in self.keys():
            if self[name]["type"] in ("observe", "sample"):
                if name not in vec_batch_nodes_dict:
                    self[name]["log_pdf"] = self[name]["fn"].log_pdf(
                        self[name]["value"],
                        *self[name]["args"][0],
                        **self[name]["args"][1]) * self[name]["scale"]
                    log_p += self[name]["log_pdf"]
                else:
                    self[name]["batch_log_pdf"] = self[name]["fn"].batch_log_pdf(
                        self[name]["value"],
                        *self[name]["args"][0],
                        **self[name]["args"][1]) * self[name]["scale"]
                    log_p += self[name]["batch_log_pdf"].sum()
        return log_p

    def batch_log_pdf(self):
        """
        Compute the local and overall log-probabilities of the trace
        """
        log_p = 0.0
        for name in self.keys():
            if self[name]["type"] in ("observe", "sample"):
                self[name]["batch_log_pdf"] = self[name]["fn"].batch_log_pdf(
                    self[name]["value"],
                    *self[name]["args"][0],
                    **self[name]["args"][1]) * self[name]["scale"]
                log_p += self[name]["batch_log_pdf"]
        return log_p


class TraceGraph(object):
    """
    -- encapsulates the forward graph as well as the trace of a stochastic function,
       along with some helper functions to access different node types.
    -- returned by TraceGraphPoutine
    -- visualization handled by save_visualization()
    """

    def __init__(self, G, trace, stochastic_nodes, reparameterized_nodes,
                 observation_nodes,
                 vectorized_map_data_info):
        self.G = G
        self.trace = trace
        self.reparameterized_nodes = reparameterized_nodes
        self.stochastic_nodes = stochastic_nodes
        self.nonreparam_stochastic_nodes = list(set(stochastic_nodes) - set(reparameterized_nodes))
        self.observation_nodes = observation_nodes
        self.vectorized_map_data_info = vectorized_map_data_info

    def get_stochastic_nodes(self):
        """
        get all sample nodes in graph
        """
        return self.stochastic_nodes

    def get_nonreparam_stochastic_nodes(self):
        """
        get all non-reparameterized sample nodes in graph
        """
        return self.nonreparam_stochastic_nodes

    def get_reparam_stochastic_nodes(self):
        """
        get all reparameterized sample nodes in graph
        """
        return self.reparameterized_nodes

    def get_nodes(self):
        """
        get all nodes in graph
        """
        return self.G.nodes()

    def get_children(self, node, with_self=False):
        """
        get children of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the children
        """
        children = self.G.successors(node)
        if with_self:
            children.append(node)
        return children

    def get_parents(self, node, with_self=False):
        """
        get parents of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the parents
        """
        parents = self.G.predecessors(node)
        if with_self:
            parents.append(node)
        return parents

    def get_ancestors(self, node, with_self=False):
        """
        get ancestors of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the ancestors
        """
        ancestors = list(networkx.ancestors(self.G, node))
        if with_self:
            ancestors.append(node)
        return ancestors

    def get_descendants(self, node, with_self=False):
        """
        get descendants of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the descendants
        """
        descendants = list(networkx.descendants(self.G, node))
        if with_self:
            descendants.append(node)
        return descendants

    def get_trace(self):
        """
        get the Trace associated with the TraceGraph
        """
        return self.trace

    def get_graph(self):
        """
        get the graph associated with the TraceGraph
        """
        return self.G

    def save_visualization(self, graph_output):
        """
        render graph and save to file
        :param graph_output: the graph will be saved to graph_output.pdf
        -- non-reparameterized stochastic nodes are salmon
        -- reparameterized stochastic nodes are half salmon, half grey
        -- observation nodes are green
        """
        g = graphviz.Digraph()
        for label in self.G.nodes():
            shape = 'ellipse'
            if label in self.stochastic_nodes and label not in self.reparameterized_nodes:
                fillcolor = 'salmon'
            elif label in self.reparameterized_nodes:
                fillcolor = 'lightgrey;.5:salmon'
            elif label in self.observation_nodes:
                fillcolor = 'darkolivegreen3'
            else:
                fillcolor = 'grey'
            g.node(label, label=label, shape=shape, style='filled', fillcolor=fillcolor)

        for label1, label2 in self.G.edges():
            g.edge(label1, label2)

        g.render(graph_output, view=False, cleanup=True)

