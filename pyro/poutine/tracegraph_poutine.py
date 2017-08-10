import pyro
import torch
from torch.autograd import Variable, Function
import graphviz
import networkx
from collections import defaultdict
from .trace_poutine import TracePoutine


def varid(v):  # XXX what's the best way to do this??
    return int(str(v.data._cdata) + str(v.data.__hash__()))


class TraceGraph(object):
    """
    -- encapsulates the forward graph as well as the trace of a stochastic function,
       along with some helper functions to access different node types.
    -- returned by TraceGraphPoutine
    """
    def __init__(self, G, trace, stochastic_nodes, reparameterized_nodes, param_nodes):
        self.G = G
        self.trace = trace
        self.param_nodes = param_nodes
        self.reparameterized_nodes = reparameterized_nodes
        self.stochastic_nodes = stochastic_nodes
        self.nonreparam_stochastic_nodes = list(set(stochastic_nodes) - set(reparameterized_nodes))

    def get_stochastic_nodes(self):
        return self.stochastic_nodes

    def get_nonreparam_stochastic_nodes(self):
        return self.nonreparam_stochastic_nodes

    def get_reparam_stochastic_nodes(self):
        return self.reparameterized_nodes

    def get_nodes(self):
        return self.G.nodes()

    def get_children(self, node, with_self=False):
        children = self.G.successors(node)
        if with_self:
            children.append(node)
        return children

    def get_parents(self, node, with_self=False):
        parents = self.G.predecessors(node)
        if with_self:
            parents.append(node)
        return parents

    def get_ancestors(self, node, with_self=False):
        ancestors = list(networkx.ancestors(self.G, node))
        if with_self:
            ancestors.append(node)
        return ancestors

    def get_descendants(self, node, with_self=False):
        descendants = list(networkx.descendants(self.G, node))
        if with_self:
            descendants.append(node)
        return descendants

    def get_trace(self):
        return self.trace


class TraceGraphPoutine(TracePoutine):
    """
    trace graph poutine (with optional visualization)
    -- if graph_output = None, no visualization is made
    -- parameter nodes are light blue
    -- non-reparameterized stochastic nodes are salmon
    -- reparameterized stochastic nodes are half salmon, half grey
    -- observation nodes are green
    -- intermediate nodes are grey
    -- include_intermediates controls granularity of visualization
    -- if there's a return value node, it's visualized as a double circle
    XXX some things are still funky with the graph (it's not necessarily a DAG) although i don't
        think this affects TraceGraph_KL_QP. this has to do with the unique id used, how that
        interacts with operations like torch.view(), etc. all this should be solved once we
        move away from monkeypatching?
    XXX graph and graph visualization always contains parameters. optionally remove and/or
        always remove for tracegraph_klqp?
    XXX seems to play somewhat strangely with replay?
    """
    def __init__(self, fn, graph_output=None, include_intermediates=False):
        super(TraceGraphPoutine, self).__init__(fn)
        self.graph_output = graph_output
        self.include_intermediates = include_intermediates

    def _enter_poutine(self, *args, **kwargs):
        """
        enter, monkeypatch Function.__call__, and set up data structures
        """
        super(TraceGraphPoutine, self)._enter_poutine(*args, **kwargs)
        self.old_function__call__ = Function.__call__
        self.monkeypatch_active = True
        self.stochastic_nodes = []
        self.reparameterized_nodes = []
        self.param_nodes = []
        self.observation_nodes = []
        self.G = networkx.DiGraph()
        self.id_to_name_dict = {}

        # this function wraps pytorch computations so that we can
        # construct the forward graph
        def new_function__call__(func, *args, **kwargs):
            output = self.old_function__call__(func, *args, **kwargs)
            if self.monkeypatch_active:
                inputs = [a for a in args if isinstance(a, Variable)]
                inputs += [a for a in kwargs.values() if isinstance(a, Variable)]
                self.register_function(inputs, func, output)
            return output

        Function.__call__ = new_function__call__

    def register_function(self, inputs, creator, output):
        """
        register inputs and output of pytorch function with graph
        """
        assert type(output) not in [tuple, list, dict],\
            "registor_function: output type not as expected"
        output_id = varid(output)
        for _input in inputs:
            input_id = varid(_input)
            self.G.add_edge(input_id, output_id)

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Register the return value from the function on exit and make visualization
        Return a TraceGraph object that contains the forward graph and trace
        """
        Function.__call__ = self.old_function__call__
        self.trace = super(TraceGraphPoutine, self)._exit_poutine(ret_val, *args, **kwargs)

        # register return value
        if ret_val is not None:
            self.ret_val = varid(ret_val)
            if self.ret_val not in self.id_to_name_dict:
                self.id_to_name_dict[self.ret_val] = 'return'

        if not self.include_intermediates:
            self.remove_intermediates()

        # remove loops
        # for edge in self.G.edges():
        #    if edge[0]==edge[1]:
        #        self.G.remove_edge(*edge)

        if self.graph_output is not None:
            self.save_visualization()

        # in any case we remove intermediates from the graph passed to TraceGraph
        if self.include_intermediates:
            self.remove_intermediates()

        return TraceGraph(networkx.relabel_nodes(self.G, self.id_to_name_dict), self.trace,
                          self.stochastic_nodes, self.reparameterized_nodes,
                          self.param_nodes)

    def remove_intermediates(self):
        """
        remove unnamed intermediates from graph
        """

        for node in self.G.nodes():
            if node not in self.id_to_name_dict:
                children = self.G.successors(node)
                parents = self.G.predecessors(node)
                for p in parents:
                    for c in children:
                        if c != p:
                            self.G.add_edge(p, c)
                self.G.remove_node(node)

    def save_visualization(self):
        """
        render graph and save to graph_output
        """
        g = graphviz.Digraph()
        for vid in self.G.nodes():
            if vid in self.id_to_name_dict:
                label = self.id_to_name_dict[vid]
                shape = 'ellipse' if vid != self.ret_val else 'doublecircle'
                if label in self.param_nodes:
                    fillcolor = 'lightblue'
                elif label in self.stochastic_nodes and label not in self.reparameterized_nodes:
                    fillcolor = 'salmon'
                elif label in self.reparameterized_nodes:
                    fillcolor = 'lightgrey;.5:salmon'
                elif label in self.observation_nodes:
                    fillcolor = 'darkolivegreen3'
                else:
                    fillcolor = 'lightgrey'
                g.node(str(vid), label=label, shape=shape, style='filled', fillcolor=fillcolor)
            else:
                label = 'unnamed\nintermediate'
                g.node(str(vid), label=label, shape='ellipse', style='filled')

        for vid1, vid2 in self.G.edges():
            g.edge(str(vid1), str(vid2))

        g.render(self.graph_output, view=False, cleanup=True)

    def _pyro_sample(self, prev_val, name, dist, *args, **kwargs):
        """
        register sampled variable for graph construction
        """
        self.monkeypatch_active = False
        val = super(TraceGraphPoutine, self)._pyro_sample(prev_val, name, dist,
                                                          *args, **kwargs)
        self.monkeypatch_active = True

        self.id_to_name_dict[varid(val)] = name
        for arg in args:
            if isinstance(arg, Variable):
                self.G.add_edge(varid(arg), varid(val))
        self.stochastic_nodes.append(name)
        if dist.reparameterized:
            self.reparameterized_nodes.append(name)
        return val

    def _pyro_param(self, prev_val, name, *args, **kwargs):
        """
        register parameter for graph construction
        """
        retrieved = super(TraceGraphPoutine, self)._pyro_param(prev_val, name,
                                                               *args, **kwargs)
        self.id_to_name_dict[varid(retrieved)] = name
        self.param_nodes.append(name)
        return retrieved

    def _pyro_observe(self, prev_val, name, fn, obs, *args, **kwargs):
        """
        register observe dependencies for graph construction
        """
        self.monkeypatch_active = False
        val = super(TraceGraphPoutine, self)._pyro_observe(prev_val, name, fn, obs,
                                                           *args, **kwargs)
        self.monkeypatch_active = True
        self.id_to_name_dict[varid(val)] = name
        self.observation_nodes.append(name)
        for arg in args:
            if isinstance(arg, Variable):
                self.G.add_edge(varid(arg), varid(val))
        return val
