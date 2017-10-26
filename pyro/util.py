import pyro
import graphviz
import numpy as np
import functools
import warnings
import torch
from torch.autograd import Variable
from torch.nn import Parameter


def detach_iterable(iterable):
    if isinstance(iterable, Variable):
        return iterable.detach()
    else:
        return [var.detach() for var in iterable]


def _dict_to_tuple(d):
    """
    Recursively converts a dictionary to a list of key-value tuples
    Only intended for use as a helper function inside memoize!!
    May break when keys cant be sorted, but that is not an expected use-case
    """
    if isinstance(d, dict):
        return tuple([(k, _dict_to_tuple(d[k])) for k in sorted(d.keys())])
    else:
        return d


def get_tensor_data(t):
    if isinstance(t, Variable):
        return t.data
    return t


def memoize(fn):
    """
    https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
    unbounded memoize
    alternate in py3: https://docs.python.org/3/library/functools.html
    lru_cache
    """
    mem = {}

    def _fn(*args, **kwargs):
        kwargs_tuple = _dict_to_tuple(kwargs)
        if (args, kwargs_tuple) not in mem:
            mem[(args, kwargs_tuple)] = fn(*args, **kwargs)
        return mem[(args, kwargs_tuple)]
    return _fn


def set_rng_seed(rng_seed):
    """
    Sets seeds of torch, numpy, and torch.cuda (if available).
    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    np.random.seed(rng_seed)


def ones(*args, **kwargs):
    """
    :param torch.Tensor type_as: optional argument for tensor type

    A convenience function for Parameter(torch.ones(...))
    """
    retype = kwargs.pop('type_as', None)
    p_tensor = torch.ones(*args, **kwargs)
    return Parameter(p_tensor if retype is None else p_tensor.type_as(retype))


def zeros(*args, **kwargs):
    """
    :param torch.Tensor type_as: optional argument for tensor type

    A convenience function for Parameter(torch.zeros(...))
    """
    retype = kwargs.pop('type_as', None)
    p_tensor = torch.zeros(*args, **kwargs)
    return Parameter(p_tensor if retype is None else p_tensor.type_as(retype))


def ng_ones(*args, **kwargs):
    """
    :param torch.Tensor type_as: optional argument for tensor type

    A convenience function for Variable(torch.ones(...), requires_grad=False)
    """
    retype = kwargs.pop('type_as', None)
    p_tensor = torch.ones(*args, **kwargs)
    return Variable(p_tensor if retype is None else p_tensor.type_as(retype), requires_grad=False)


def ng_zeros(*args, **kwargs):
    """
    :param torch.Tensor type_as: optional argument for tensor type

    A convenience function for Variable(torch.ones(...), requires_grad=False)
    """
    retype = kwargs.pop('type_as', None)
    p_tensor = torch.zeros(*args, **kwargs)
    return Variable(p_tensor if retype is None else p_tensor.type_as(retype), requires_grad=False)


def log_sum_exp(vecs):
    n = len(vecs.size())
    if n == 1:
        vecs = vecs.view(1, -1)
    _, idx = torch.max(vecs, 1)
    max_score = torch.index_select(vecs, 1, idx.view(-1))
    ret = max_score + torch.log(torch.sum(torch.exp(vecs - max_score.expand_as(vecs))))
    if n == 1:
        return ret.view(-1)
    return ret


def zero_grads(tensors):
    """
    Sets gradients of list of Variables to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            if p.grad.volatile:
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


def tensor_histogram(ps, vs):
    """
    make a histogram from weighted Variable/Tensor/ndarray samples
    Horribly slow...
    """
    # first, get everything into the same form: numpy arrays
    np_vs = []
    for v in vs:
        _v = v
        if isinstance(_v, Variable):
            _v = _v.data
        if isinstance(_v, torch.Tensor):
            _v = _v.cpu().numpy()
        np_vs.append(_v)
    # now form the histogram
    hist = dict()
    for p, v, np_v in zip(ps, vs, np_vs):
        k = tuple(np_v.flatten().tolist())
        if k not in hist:
            # XXX should clone?
            hist[k] = [0.0, v]
        hist[k][0] = hist[k][0] + p
    # now split into keys and original values
    ps2 = []
    vs2 = []
    for k in hist.keys():
        ps2.append(hist[k][0])
        vs2.append(hist[k][1])
    # return dict suitable for passing into Categorical
    return {"ps": torch.cat(ps2), "vs": np.array(vs2).flatten()}


def basic_histogram(ps, vs):
    """
    make a histogram from weighted things that aren't tensors
    Horribly slow...
    """
    assert isinstance(vs, (list, tuple)), \
        "vs must be a primitive type that preserves ordering at construction"
    hist = {}
    for i, v in enumerate(vs):
        if v not in hist:
            hist[v] = 0.0
        hist[v] = hist[v] + ps[i]
    return {"ps": torch.cat([hist[v] for v in hist.keys()]),
            "vs": [v for v in hist.keys()]}


def apply_stack(initial_msg):
    """
    :param dict initial_msg: the starting version of the trace site
    :returns: an updated message that is the final version of the trace site

    Execute the poutine stack at a single site according to the following scheme:
    1. Walk down the stack from top to bottom, collecting into the message
        all information necessary to execute the stack at that site
    2. For each poutine in the stack from bottom to top:
           Execute the poutine with the message;
           If the message field "stop" is True, stop;
           Otherwise, continue
    3. Return the updated message
    """
    stack = pyro._PYRO_STACK
    # TODO check at runtime if stack is valid

    # msg is used to pass information up and down the stack
    msg = initial_msg

    # first, gather all information necessary to apply the stack to this site
    for frame in reversed(stack):
        msg = frame._prepare_site(msg)

    # go until time to stop?
    for frame in stack:
        assert msg["type"] in ("sample", "param"), \
            "{} is an invalid site type, how did that get there?".format(msg["type"])

        msg["value"] = getattr(frame, "_pyro_{}".format(msg["type"]))(msg)

        if msg["stop"]:
            break

    return msg


class NonlocalExit(Exception):
    """
    Exception for exiting nonlocally from poutine execution.

    Used by poutine.EscapePoutine to return site information.
    """
    def __init__(self, site, *args, **kwargs):
        """
        :param site: message at a pyro site

        constructor.  Just stores the input site.
        """
        super(NonlocalExit, self).__init__(*args, **kwargs)
        self.site = site


def enum_extend(trace, msg, num_samples=None):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :param num_samples: maximum number of extended traces to return.
    :returns: a list of traces, copies of input trace with one extra site

    Utility function to copy and extend a trace with sites based on the input site
    whose values are enumerated from the support of the input site's distribution.

    Used for exact inference and integrating out discrete variables.
    """
    if num_samples is None:
        num_samples = -1

    extended_traces = []
    for i, s in enumerate(msg["fn"].support(*msg["args"], **msg["kwargs"])):
        if i > num_samples and num_samples >= 0:
            break
        msg_copy = msg.copy()
        msg_copy.update(value=s)
        tr_cp = trace.copy()
        tr_cp.add_node(msg["name"], **msg_copy)
        extended_traces.append(tr_cp)
    return extended_traces


def mc_extend(trace, msg, num_samples=None):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :param num_samples: maximum number of extended traces to return.
    :returns: a list of traces, copies of input trace with one extra site

    Utility function to copy and extend a trace with sites based on the input site
    whose values are sampled from the input site's function.

    Used for Monte Carlo marginalization of individual sample sites.
    """
    if num_samples is None:
        num_samples = 1

    extended_traces = []
    for i in range(num_samples):
        msg_copy = msg.copy()
        msg_copy["value"] = msg_copy["fn"](*msg_copy["args"], **msg_copy["kwargs"])
        tr_cp = trace.copy()
        tr_cp.add_node(msg_copy["name"], **msg_copy)
        extended_traces.append(tr_cp)
    return extended_traces


def discrete_escape(trace, msg):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :returns: boolean decision value

    Utility function that checks if a sample site is discrete and not already in a trace.

    Used by EscapePoutine to decide whether to do a nonlocal exit at a site.
    Subroutine for integrating out discrete variables for variance reduction.
    """
    return (msg["type"] == "sample") and \
        (not msg["is_observed"]) and \
        (msg["name"] not in trace) and \
        (getattr(msg["fn"], "enumerable", False))


def all_escape(trace, msg):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :returns: boolean decision value

    Utility function that checks if a site is not already in a trace.

    Used by EscapePoutine to decide whether to do a nonlocal exit at a site.
    Subroutine for approximately integrating out variables for variance reduction.
    """
    return (msg["type"] == "sample") and \
        (not msg["is_observed"]) and \
        (msg["name"] not in trace)


def save_visualization(trace, graph_output):
    """
    :param pyro.poutine.Trace trace: a trace to be visualized
    :param graph_output: the graph will be saved to graph_output.pdf
    :type graph_output: str

    Take a trace generated by poutine.trace with `graph_type='dense'` and render
    the graph with the output saved to file.

    - non-reparameterized stochastic nodes are salmon
    - reparameterized stochastic nodes are half salmon, half grey
    - observation nodes are green

    Example:

    trace = pyro.poutine.trace(model, graph_type="dense").get_trace()
    save_visualization(trace, 'output')
    """
    g = graphviz.Digraph()

    for label in trace.nodes:
        shape = 'ellipse'
        if label in trace.stochastic_nodes and label not in trace.reparameterized_nodes:
            fillcolor = 'salmon'
        elif label in trace.reparameterized_nodes:
            fillcolor = 'lightgrey;.5:salmon'
        elif label in trace.observation_nodes:
            fillcolor = 'darkolivegreen3'
        else:
            # only visualize RVs
            continue
        g.node(label, label=label, shape=shape, style='filled', fillcolor=fillcolor)

    for label1, label2 in trace.edges:
        g.edge(label1, label2)

    g.render(graph_output, view=False, cleanup=True)


def check_site_names(model_trace, guide_trace):
    """
    :param pyro.poutine.Trace model_trace: Trace object of the model
    :param pyro.poutine.Trace guide_trace: Trace object of the guide
    :raises: RuntimeWarning

    Checks that (1) there is a bijection between the samples in the guide
    and the samples in the model, and (2) each `iarange` statement in the
    guide also appears in the model.
    """
    # Check ordinary sample sites.
    model_vars = set(name for name, site in model_trace.nodes.items()
                     if site["type"] == "sample" and not site["is_observed"]
                     if type(site["fn"]).__name__ != "_Subsample")
    guide_vars = set(name for name, site in guide_trace.nodes.items()
                     if site["type"] == "sample"
                     if type(site["fn"]).__name__ != "_Subsample")
    if not (guide_vars <= model_vars):
        warnings.warn("Found vars in guide but not model: {}".format(guide_vars - model_vars))
    if not (model_vars <= guide_vars):
        warnings.warn("Found vars in model but not guide: {}".format(model_vars - guide_vars))

    # Check subsample sites introduced by iarange.
    model_vars = set(name for name, site in model_trace.nodes.items()
                     if site["type"] == "sample" and not site["is_observed"]
                     if type(site["fn"]).__name__ == "_Subsample")
    guide_vars = set(name for name, site in guide_trace.nodes.items()
                     if site["type"] == "sample"
                     if type(site["fn"]).__name__ == "_Subsample")
    if not (guide_vars <= model_vars):
        warnings.warn("Found iarange statements in guide but not model: {}".format(guide_vars - model_vars))


def deep_getattr(obj, name):
    """
    Python getattr() for arbitrarily deep attributes
    Throws an AttirbuteError if bad attribute
    """
    return functools.reduce(getattr, name.split("."), obj)
