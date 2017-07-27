import pyro
import torch
from torch.autograd import Variable, Function
import graphviz
import networkx
from collections import defaultdict
from .trace_poutine import TracePoutine


class TraceGraph(object):
    def __init__(self, G, trace, stochastic_nodes, reparameterized_nodes, param_nodes,
                 observe_stochastic_parents):
        self.G = G
        self.trace = trace
        self.param_nodes = param_nodes
        self.reparameterized_nodes = reparameterized_nodes
        self.stochastic_nodes = stochastic_nodes
        self.nonreparam_stochastic_nodes = list(set(stochastic_nodes) - set(reparameterized_nodes))
        self.observe_stochastic_parents = observe_stochastic_parents

    def get_direct_stochastic_children_of_parameters(self):
        direct_children = set()
        for param in self.param_nodes:
            for node in self.get_children(param):
                if node in self.nonreparam_stochastic_nodes:
                    direct_children.add(node)
        return list(direct_children)

    def get_nonreparam_stochastic_nodes(self):
        return self.nonreparam_stochastic_nodes

    def get_nodes(self):
	return self.G.nodes()

    def get_children(self, node, with_self = False):
	children = self.G.successors(node)
        if not with_self:
            return children
        children.append(node)
        return children

    def get_parents(self, node):
	return self.G.predecessors(node)

    def get_ancestors(self, node, with_self = False):
        if with_self and node not in self.G.nodes():
            return node
	ancestors = list(networkx.ancestors(self.G, node))
        if with_self:
            ancestors.append(node)
        return ancestors

    def get_descendants(self, node, with_self = False):
	descendants = list(networkx.descendants(self.G, node))
        if not with_self:
            return descendants
        descendants.append(node)
        return descendants

    def get_trace(self):
        return self.trace

class TraceGraphPoutine(TracePoutine):
    """
    trace graph poutine (with optional visualization)
    -- if graph_output = None, no visualization is made
    -- skip_creators and include_intermediates control the granularity of the
       constructed graph
    """
    def __init__(self, fn, graph_output = None, skip_creators = True, include_intermediates = False):
        super(TraceGraphPoutine, self).__init__(fn)
        self.graph_output = graph_output
        self.skip_creators = skip_creators
        self.include_intermediates = include_intermediates
        if not self.skip_creators:
            assert(self.include_intermediates),\
                "TraceGraphPoutine: cannot include creators and not include intermediates"

    def _enter_poutine(self, *args, **kwargs):
        """
        enter, monkeypatch Function.__call__, and set up data structures
        """
        super(TraceGraphPoutine, self)._enter_poutine(*args, **kwargs)
        self.old_function__call__ = Function.__call__
        self.var_to_name_dict = {}
        self.var_trace = defaultdict(lambda: {})
        self.func_trace = defaultdict(lambda: {})
	self.variables = {}
	self.funcs = {}
        self.output_to_input = {}
        self.stochastic_nodes = []
        self.reparameterized_nodes = []
        self.param_nodes = []
        self.detached_var_dict = {}
        self.sample_args = {}
        self.observe_args = {}
        self.observe_stochastic_parents = defaultdict(lambda: [])

	def new_function__call__(func, *args, **kwargs):
	    inputs =  [a for a in args            if isinstance(a, Variable)]
	    inputs += [a for a in kwargs.values() if isinstance(a, Variable)]
	    output = self.old_function__call__(func, *args, **kwargs)
	    self.register_creator(inputs, func, output)
	    return output

	Function.__call__ = new_function__call__

    def register_creator(self, inputs, creator, output):
	cid = id(creator)
	oid = id(output)
	if oid in self.variables:
            assert False, "registor_creator: something went wrong with variable list"
	# connect creator to input
	for _input in inputs:
	    iid = id(_input)
	    self.func_trace[cid][iid] = _input
	    # register input
	    self.variables[iid] = _input
	# connect output to creator
        assert type(output) not in [tuple, list, dict], "registor_creator: output type not as expected"
	self.var_trace[oid][cid] = creator
	# register creator and output and all inputs
	self.variables[oid] = output
	self.funcs[cid] = creator
        # connect output directly to inputs
	self.output_to_input[oid] = [ id(_input) for _input in inputs ]

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Register the return value from the function on exit and make visualization
        """
        self.trace = super(TraceGraphPoutine, self)._exit_poutine(ret_val, *args, **kwargs)

	Function.__call__ = self.old_function__call__

        if ret_val not in self.var_to_name_dict:
            self.var_to_name_dict[ret_val] = 'return'
        self.ret_val = str(id(ret_val))

        self.create_graph()
        if self.graph_output is not None:
            self.save_visualization()

        if self.skip_creators and not self.include_intermediates:
            return TraceGraph(networkx.relabel_nodes(self.G, self.vid_to_names), self.trace,
                              self.stochastic_nodes, self.reparameterized_nodes,
                              self.param_nodes, self.observe_stochastic_parents)
        else:
            return self.G

    def create_graph(self):
        self.G = networkx.DiGraph()
        self.vid_to_names = {str(k): self.var_to_name_dict[v] for k,v in self.variables.iteritems()
                             if v in self.var_to_name_dict}
	# add variable nodes
	for vid, var in self.variables.iteritems():
	    if isinstance(var, Variable):
                self.G.add_node(str(vid))
	    else:
		assert False, var.__class__
        for node in self.sample_args:
            for arg in self.sample_args[node]:
                self.G.add_edge(str(arg), str(node))
	# add creator nodes
        if not self.skip_creators:
            for cid in self.func_trace:
                self.G.add_node(str(cid))
            # add edges between creator and inputs
            for cid in self.func_trace:
                for iid in self.func_trace[cid]:
                    self.G.add_edge(str(iid), str(cid))
            # add edges between outputs and creators
            for oid in self.var_trace:
                for cid in self.var_trace[oid]:
                    self.G.add_edge(str(cid), str(oid))
        else:
            for oid in self.output_to_input:
                for iid in self.output_to_input[oid]:
                    self.G.add_edge(str(iid), str(oid))

        for node in self.detached_var_dict:
            if self.ret_val == node:
                self.ret_val = self.detached_var_dict[node]
            if node in self.G.nodes():
                self.G = networkx.contracted_nodes(self.G, self.detached_var_dict[node], node)

        self.vid_to_names2 = {str(id(k)): self.var_to_name_dict[k] for k,v in self.var_to_name_dict.items()}
        for key in self.observe_args:
            for arg in self.observe_args[key]:
                ancestors = [str(arg)]
                if str(arg) in self.G.nodes():
                    ancestors.extend(networkx.ancestors(self.G, str(arg)))
                ancestors2 = [a if a not in self.detached_var_dict \
                                else self.detached_var_dict[a] for a in ancestors]
                for node in self.stochastic_nodes:
                    for a in ancestors2:
                        if a in self.vid_to_names2 and self.vid_to_names2[a] == node:
                            self.observe_stochastic_parents[key].append(node)

        if not self.include_intermediates:
            for vid1, var1 in self.variables.iteritems():
                for vid2, var2 in self.variables.iteritems():
                    if vid1==vid2:
                        continue
                    if var1 in self.var_to_name_dict and var2 in self.var_to_name_dict:
                        try:
                            path = networkx.astar_path(self.G, str(vid1), str(vid2))
                            #print "\npath[%s,%s]" % (self.var_to_name_dict[var1],
                            #                       self.var_to_name_dict[var2]) #, path
                            connect = True
                            for edge in path[1:-1]:
                                if edge in self.vid_to_names:
                                    connect = False
                            if connect:
                                self.G.add_edge(str(vid1), str(vid2))

                        except Exception as e:
                            pass

            for vid in self.G.nodes():
                if vid not in self.vid_to_names and int(vid) not in self.funcs:
                    self.G.remove_node(vid)

    def save_visualization(self):
	g = graphviz.Digraph()
        for vid in self.G.nodes():
            if vid in self.vid_to_names:
                label = self.vid_to_names[vid]
                shape = 'ellipse' if not vid==self.ret_val else 'doublecircle'
                fillcolor = 'lightblue' if label not in self.stochastic_nodes else 'salmon'
                if label in self.reparameterized_nodes:
                    fillcolor = 'lightgrey;.5:salmon'
                g.node(vid, label=label, shape=shape, style='filled', fillcolor=fillcolor)
            elif int(vid) in self.funcs:
                g.node(vid, label=str(self.funcs[int(vid)].__class__.__name__), shape='rectangle',
                                 style='filled', fillcolor='orange')
            elif self.include_intermediates:
                label = 'unnamed\nintermediate'
                g.node(vid, label=label, shape='ellipse', style='filled')

        for vid1, vid2 in self.G.edges():
            if self.include_intermediates or (vid1 in self.vid_to_names and\
                                              vid2 in self.vid_to_names):
                g.edge(vid1, vid2)

	g.render(self.graph_output, view=False, cleanup=True)

    def _pyro_sample(self, prev_val, name, dist, *args, **kwargs):
        """
        register sampled variable for graph construction
        """
        val = super(TraceGraphPoutine, self)._pyro_sample(prev_val, name, dist,
                                                     *args, **kwargs)
        self.var_to_name_dict[val] = name
        self.stochastic_nodes.append(name)
        if dist.reparameterized:
            self.reparameterized_nodes.append(name)
        if not dist.reparameterized:
            self.sample_args[id(val)] = [id(a) for a in args]
            self.variables[id(val)] = val
            val_detach = Variable(val.data)
            self.detached_var_dict[str(id(val_detach))] = str(id(val))
            return val_detach
        return val

    def _pyro_param(self, prev_val, name, *args, **kwargs):
        """
        register parameter for graph construction
        """
        retrieved = super(TraceGraphPoutine, self)._pyro_param(prev_val, name,
                                                          *args, **kwargs)
        self.var_to_name_dict[retrieved] = name
        self.param_nodes.append(name)
        return retrieved

    def _pyro_observe(self, prev_val, name, fn, obs, *args, **kwargs):
        """
        register observe dependencies
        """
        val = super(TraceGraphPoutine, self)._pyro_observe(prev_val, name, fn, obs,
                                                      *args, **kwargs)
        self.observe_args[name] = [id(a) for a in args]
        return val
