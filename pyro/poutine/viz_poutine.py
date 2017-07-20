import pyro
import torch
from torch.autograd import Variable, Function
import graphviz
import networkx
from collections import defaultdict
from .poutine import Poutine


class VizPoutine(Poutine):
    """
    visualization.
    """
    def __init__(self, fn, output_file, skip_creators = False, include_intermediates = True):
        super(VizPoutine, self).__init__(fn)
        self.output_file = output_file
        self.skip_creators = skip_creators
        self.include_intermediates = include_intermediates
        if not self.skip_creators:
            assert(self.include_intermediates),\
                "VizPoutine: cannot include creators and not include intermediates"

    def _enter_poutine(self, *args, **kwargs):
        """
        enter, monkeypatch Function.__call__, and set up data structures
        """
        super(VizPoutine, self)._enter_poutine(*args, **kwargs)
        self.old_function__call__ = Function.__call__
        self.var_to_name_dict = {}
        self.var_trace = defaultdict(lambda: {})
        self.func_trace = defaultdict(lambda: {})
	self.variables = {}
	self.funcs = {}
        self.output_to_input = {}
        self.G = networkx.DiGraph()

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
	    return
	# connect creator to input
	for _input in inputs:
	    iid = id(_input)
	    self.func_trace[cid][iid] = _input
	    # register input
	    self.variables[iid] = _input
	# connect output to creator
	assert type(output) not in [tuple, list, dict]
	self.var_trace[oid][cid] = creator
	# register creator and output and all inputs
	self.variables[oid] = output
	self.funcs[cid] = creator
        # connect output directly to inputs
	self.output_to_input[oid] = []
	for _input in inputs:
	    iid = id(_input)
	    self.output_to_input[oid].append(iid)

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Register the return value from the function on exit and make visualization
        """
        if ret_val in self.var_to_name_dict:
            self.var_to_name_dict[ret_val] += ' [return]'
        else:
            self.var_to_name_dict[ret_val] = 'return'
	Function.__call__ = self.old_function__call__
        self._create_graph()
	self.save_visualization()

    def _create_graph(self):
        self.G = networkx.DiGraph()
        self.vid_to_names = {str(k): self.var_to_name_dict[v] for k,v in self.variables.iteritems()
                             if v in self.var_to_name_dict}
	# add variable nodes
	for vid, var in self.variables.iteritems():
	    if isinstance(var, Variable):
		if var in self.var_to_name_dict:
                    self.G.add_node(str(vid))
		else:
                    self.G.add_node(str(vid))
	    else:
		assert False, var.__class__
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

    def save_visualization(self):
	g = graphviz.Digraph()
        for vid in self.G.nodes():
            if vid in self.vid_to_names:
                label = self.vid_to_names[vid]
                g.node(vid, label=label, shape='ellipse', style='filled',
                       fillcolor='lightblue')
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

	g.render(self.output_file, view=False)

    def _pyro_sample(self, prev_val, name, dist, *args, **kwargs):
        """
        TODO
        """
        val = super(VizPoutine, self)._pyro_sample(prev_val, name, dist,
                                                     *args, **kwargs)
        self.var_to_name_dict[val] = name
        return val

    def _pyro_param(self, prev_val, name, *args, **kwargs):
        """
        TODO
        """
        retrieved = super(VizPoutine, self)._pyro_param(prev_val, name,
                                                          *args, **kwargs)
        self.var_to_name_dict[retrieved] = name
        return retrieved
