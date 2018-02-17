from __future__ import absolute_import, division, print_function

import functools
import numbers
import warnings

import graphviz
import numpy as np

import torch
from pyro.params import _PYRO_PARAM_STORE

from pyro.poutine.poutine import _PYRO_STACK
from pyro.poutine.util import site_is_subsample
from pyro.shim import is_volatile
from torch.autograd import Variable
from torch.nn import Parameter


def validate_message(msg):
    """
    Asserts that the message has a valid format.
    :returns: None
    """
    assert msg["type"] in ("sample", "param"), \
        "{} is an invalid site type, how did that get there?".format(msg["type"])


def default_process_message(msg):
    """
    Default method for processing messages in inference.
    :param msg: a message to be processed
    :returns: None
    """
    validate_message(msg)
    if msg["type"] == "sample":
        fn, args, kwargs = \
            msg["fn"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg

        if msg["is_observed"]:
            assert msg["value"] is not None
            val = msg["value"]
        else:
            val = fn(*args, **kwargs)

        # after fn has been called, update msg to prevent it from being called again.
        msg["done"] = True
        msg["value"] = val
    elif msg["type"] == "param":
        name, args, kwargs = \
            msg["name"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg

        ret = _PYRO_PARAM_STORE.get_param(name, *args, **kwargs)

        # after the param store has been queried, update msg["done"]
        # to prevent it from being queried again.
        msg["done"] = True
        msg["value"] = ret
    else:
        assert False
    return None


def am_i_wrapped():
    """
    Checks whether the current computation is wrapped in a poutine.
    :returns: bool
    """
    return len(_PYRO_STACK) > 0


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


def is_nan(x):
    """
    A convenient function to check if a Tensor contains all nan; also works with numbers
    and torch.autograd.Variable
    """
    if isinstance(x, numbers.Number):
        return x != x
    return (x != x).all()


def is_inf(x):
    """
    A convenient function to check if a Tensor contains all inf; also works with numbers
    and torch.autograd.Variable
    """
    if isinstance(x, numbers.Number):
        return x == float('inf')
    return (x == float('inf')).all()


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
            if is_volatile(p.grad):
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


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
    stack = _PYRO_STACK
    # TODO check at runtime if stack is valid

    # msg is used to pass information up and down the stack
    msg = initial_msg

    counter = 0
    # go until time to stop?
    for frame in stack:
        validate_message(msg)

        counter = counter + 1

        frame._process_message(msg)

        if msg["stop"]:
            break

    default_process_message(msg)

    for frame in reversed(stack[0:counter]):
        frame._postprocess_message(msg)

    cont = msg["continuation"]
    if cont is not None:
        cont(msg)

    return None


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

    for label, node in trace.nodes.items():
        if site_is_subsample(node):
            continue
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
        if site_is_subsample(trace.nodes[label1]):
            continue
        if site_is_subsample(trace.nodes[label2]):
            continue
        g.edge(label1, label2)

    g.render(graph_output, view=False, cleanup=True)


def check_model_guide_match(model_trace, guide_trace):
    """
    :param pyro.poutine.Trace model_trace: Trace object of the model
    :param pyro.poutine.Trace guide_trace: Trace object of the guide
    :raises: RuntimeWarning, ValueError

    Checks that (1) there is a bijection between the samples in the guide
    and the samples in the model, (2) each `iarange` statement in the guide
    also appears in the model, (3) at each sample site that appears in both
    the model and guide, the model and guide agree on sample shape.
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

    # Check shapes agree.
    for name in model_vars & guide_vars:
        model_site = model_trace.nodes[name]
        guide_site = guide_trace.nodes[name]
        if hasattr(model_site["fn"], "shape") and hasattr(guide_site["fn"], "shape"):
            model_shape = model_site["fn"].shape(*model_site["args"], **model_site["kwargs"])
            guide_shape = guide_site["fn"].shape(*guide_site["args"], **guide_site["kwargs"])
            if model_shape != guide_shape:
                raise ValueError("Model and guide dims disagree at site '{}': {} vs {}".format(
                    name, model_shape, guide_shape))

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
    Throws an AttributeError if bad attribute
    """
    return functools.reduce(getattr, name.split("."), obj)
