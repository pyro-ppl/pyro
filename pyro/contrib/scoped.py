"""
Named Data Structures
---------------------

The ``pyro.contrib.named`` module is a thin syntactic layer on top of Pyro.  It
allows Pyro models to be written to look like programs with operating on Python
data structures like ``latent.x.sample_(...)``, rather than programs with
string-labeled statements like ``x = pyro.sample("x", ...)``.

This module provides three container data structures ``named.Object``,
``named.List``, and ``named.Dict``. These data structures are intended to be
nested in each other. Together they track the address of each piece of data
in each data structure, so that this address can be used as a Pyro site. For
example::

    >>> state = named.Object("state")
    >>> print(str(state))
    state

    >>> z = state.x.y.z  # z is just a placeholder.
    >>> print(str(z))
    state.x.y.z

    >>> state.xs = named.List()  # Create a contained list.
    >>> x0 = state.xs.add()
    >>> print(str(x0))
    state.xs[0]

    >>> state.ys = named.Dict()
    >>> foo = state.ys['foo']
    >>> print(str(foo))
    state.ys['foo']

These addresses can now be used inside ``sample``, ``observe`` and ``param``
statements. These named data structures even provide in-place methods that
alias Pyro statements. For example::

    >>> state = named.Object("state")
    >>> mu = state.mu.param_(Variable(torch.zeros(1), requires_grad=True))
    >>> sigma = state.sigma.param_(Variable(torch.ones(1), requires_grad=True))
    >>> z = state.z.sample_(dist.normal, mu, sigma)
    >>> state.x.observe_(dist.normal, z, mu, sigma)

For deeper examples of how these can be used in model code, see the
`Tree Data <https://github.com/uber/pyro/blob/dev/examples/contrib/named/tree_data.py>`_
and
`Mixture <https://github.com/uber/pyro/blob/dev/examples/contrib/named/mixture.py>`_
examples.

Authors: Fritz Obermeyer, Alexander Rush
"""

from __future__ import absolute_import, division, print_function

import functools
import uuid

import pyro
import pyro.infer
import pyro.contrib.named as named
import pyro.util


def _lift(fn, _namespace=None, _latent=None, _store=None):
    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        latent = named.Object(_namespace) if _latent is None else _latent
        if _store is not None:
            _store[0] = latent
        return fn(latent, *args, **kwargs)
    return decorated


@functools.wraps(pyro.condition)
def condition(fn, data=None, data_fn=None):
    def partial_condition(latent, *args, **kwargs):
        if data is not None:
            lifted_data = {}
            for key, value in data.items():
                lifted_data["{}.{}".format(latent._name, key)] = value
        else:
            site_collector = named.Object(latent._name)
            data_fn(site_collector)
            lifted_data = dict(site_collector._items())
        pyro.condition(_lift(fn, _latent=latent),
                       data=lifted_data)(*args, **kwargs)
    return partial_condition


@functools.wraps(pyro.infer.SVI)
def SVI(model, guide, *args, **kwargs):
    namespace = str(uuid.uuid4())
    model_latent = []
    guide_latent = []
    obj = pyro.infer.SVI(_lift(model, namespace, _store=model_latent),
                         _lift(guide, namespace, _store=guide_latent),
                         *args, **kwargs)
    obj._model_latent = model_latent
    obj._guide_latent = guide_latent
    obj._namespace = namespace
    return obj


@functools.wraps(pyro.infer.Importance)
def Importance(model, guide=None, *args, **kwargs):
    namespace = str(uuid.uuid4())
    model_latent = []
    guide_latent = []
    obj = pyro.infer.Importance(_lift(model, namespace, _store=model_latent),
                                _lift(guide, namespace, _store=guide_latent)
                                if guide is not None else None,
                                *args, **kwargs)
    obj._model_latent = model_latent
    obj._guide_latent = guide_latent
    obj._namespace = namespace
    return obj


@functools.wraps(pyro.infer.Search)
def Search(model, *args, **kwargs):
    namespace = str(uuid.uuid4())
    model_latent = []
    obj = pyro.infer.Search(_lift(model, namespace, _store=model_latent),
                            *args, **kwargs)
    obj._model_latent = model_latent
    obj._namespace = namespace
    return obj


@functools.wraps(pyro.infer.Marginal)
def Marginal(trace_dist, sites=None, sites_fn=None, *args, **kwargs):
    assert trace_dist._namespace is not None, \
        "To call scoped marginal, trace_dist must be scoped."

    namespace = trace_dist._namespace
    # Rewrite sites to be local to trace_dist.
    if sites is not None:
        sites = ["{}.{}".format(namespace, site)
                 for site in sites]

    if sites_fn is not None:
        site_collector = named.Object(namespace)
        sites_fn(site_collector)
        sites = [name for name, val in site_collector._items()]

    marginal = pyro.infer.Marginal(trace_dist, sites, *args, **kwargs)

    # Intercept the call function for marginal.
    def call(*args, **kwargs):
        marginal_out = marginal(*args, **kwargs)
        # Rewrite returned marginals to be an object.
        if isinstance(marginal_out, dict):
            collector = site_collector if sites_fn is not None \
                        else named.Object()
            for key, value in marginal_out.items():
                scoped_key = key[len(namespace + '.'):]
                obj = pyro.util.deep_getattr(collector, scoped_key)
                obj.set_(value)
            return collector
        else:
            return marginal_out
    return call
