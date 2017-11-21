"""Scoped Data Structures
---------------------

The ``pyro.contrib.scoped`` module is a thin syntactic layer on top of
Pyro.  It heavily uses ``pyro.contrib.named`` to re-write Pyro models to use
locally scoped variables.  Instead of writing models like ``def
model(...)`` and creating using global sampling, scoped code is always
of the form ``def model(latent, ...)`` where latent is a uniquely
namespaced object that manages the underlying variables and the
inference state.

This module provides functions that wrapped the standard Pyro
inference and conditional functions to support scoped models as
defined above. These functions work exactly the same as standard Pyro
functions but always pass in a scoped ``named.Object`` to models.
Models are expected to use these named objects for sampling,
observations, and parameters. In addition, wherever possible they
offer additional options to define conditioning, inference sites, and
return values in the form of ``named`` objects.

For deeper examples of how these can be used in model code, see the
`HMM <https://github.com/uber/pyro/blob/dev/examples/contrib/scoped/hmm.py>`_
and
`Mixture <https://github.com/uber/pyro/blob/dev/examples/contrib/named/mixture.py>`_
examples.

Authors: Alexander Rush, Fritz Obermeyer
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
    model_latent = [None]
    guide_latent = [None]
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
    model_latent = [None]
    guide_latent = [None]
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
    model_latent = [None]
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
