"""
``pyro.contrib.autoname.scoping`` contains the implementation of
:func:`pyro.contrib.autoname.scope`, a tool for automatically appending
a semantically meaningful prefix to names of sample sites.
"""
import functools

from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import apply_stack


class ScopeMessenger(Messenger):
    """
    ``ScopeMessenger`` is the implementation of :func:`pyro.contrib.autoname.scope`
    """
    def __init__(self, prefix=None, inner=None):
        super(ScopeMessenger, self).__init__()
        self.prefix = prefix
        self.inner = inner

    def __enter__(self):
        if self.prefix is None:
            raise ValueError("no prefix was provided")
        if not self.inner:
            # to accomplish adding a counter to duplicate scopes,
            # we will treat ScopeMessenger.__enter__ like a sample statement
            # so that the same mechanism that adds counters to sample names
            # can be used to add a counter to a scope name
            msg = {
                "type": "sample",
                "name": self.prefix,
                "fn": lambda: True,
                "is_observed": False,
                "args": (),
                "kwargs": {},
                "value": None,
                "infer": {},
                "scale": 1.0,
                "cond_indep_stack": (),
                "done": False,
                "stop": False,
                "continuation": None,
                "PRUNE": True  # this keeps the dummy node from appearing in the trace
            }
            apply_stack(msg)
            self.prefix = msg["name"].split("/")[-1]
        return super(ScopeMessenger, self).__enter__()

    def __call__(self, fn):
        if self.prefix is None:
            self.prefix = fn.__code__.co_name  # fn.__name__

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with type(self)(prefix=self.prefix, inner=self.inner):
                return fn(*args, **kwargs)
        return _fn

    def _pyro_sample(self, msg):
        msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return None


def scope(fn=None, prefix=None, inner=None):
    """
    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param prefix: a string to prepend to sample names (optional if ``fn`` is provided)
    :param inner: switch to determine where duplicate name counters appear
    :returns: ``fn`` decorated with a :class:`~pyro.contrib.autoname.scoping.ScopeMessenger`

    ``scope`` prepends a prefix followed by a ``/`` to the name at a Pyro sample site.
    It works much like TensorFlow's ``name_scope`` and ``variable_scope``,
    and can be used as a context manager, a decorator, or a higher-order function.

    ``scope`` is very useful for aligning compositional models with guides or data.

    Example::

        >>> @scope(prefix="a")
        ... def model():
        ...     return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "a/x" in poutine.trace(model).get_trace()


    Example::

        >>> def model():
        ...     with scope(prefix="a"):
        ...         return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "a/x" in poutine.trace(model).get_trace()

    Scopes compose as expected, with outer scopes appearing before inner scopes in names::

        >>> @scope(prefix="b")
        ... def model():
        ...     with scope(prefix="a"):
        ...         return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "b/a/x" in poutine.trace(model).get_trace()

    When used as a decorator or higher-order function,
    ``scope`` will use the name of the input function as the prefix
    if no user-specified prefix is provided.

    Example::

        >>> @scope
        ... def model():
        ...     return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "model/x" in poutine.trace(model).get_trace()
    """
    msngr = ScopeMessenger(prefix=prefix, inner=inner)
    return msngr(fn) if fn is not None else msngr
