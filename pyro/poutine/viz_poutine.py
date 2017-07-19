import pyro
import torch
from torch.autograd import Variable, Function
import graphviz
from collections import defaultdict
from .poutine import Poutine


class VizPoutine(Poutine):
    """
    visualization.
    """
    def __init__(self, fn, output_file):
        super(VizPoutine, self).__init__(fn)
        self.output_file = output_file

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

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Register the return value from the function on exit and make visualization
        """
        if ret_val in self.var_to_name_dict:
            self.var_to_name_dict[ret_val] += ' [return]'
        else:
            self.var_to_name_dict[ret_val] = 'return'
	Function.__call__ = self.old_function__call__
	self.save_visualization()

    def save_visualization(self):
        #print "self.var_to_name_dict", self.var_to_name_dict
	g = graphviz.Digraph()
	# add variable nodes
	for vid, var in self.variables.iteritems():
	    if isinstance(var, Variable):
		if var in self.var_to_name_dict:
		    label = self.var_to_name_dict[var]
		    g.node(str(vid), label=label, shape='ellipse', style='filled',
			   fillcolor='lightblue')
		else:
		    label = 'unnamed\nintermediate'
		    g.node(str(vid), label=label, shape='ellipse', style='filled')
	    else:
		assert False, var.__class__
	# add creator nodes
	for cid in self.func_trace:
	    creator = self.funcs[cid]
	    g.node(str(cid), label=str(creator.__class__.__name__), shape='rectangle',
		   style='filled', fillcolor='orange')
	# add edges between creator and inputs
	for cid in self.func_trace:
	    for iid in self.func_trace[cid]:
		g.edge(str(iid), str(cid))
	# add edges between outputs and creators
	for oid in self.var_trace:
	    for cid in self.var_trace[oid]:
		g.edge(str(cid), str(oid))
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
