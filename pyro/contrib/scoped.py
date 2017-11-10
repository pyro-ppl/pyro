from __future__ import absolute_import, division, print_function

import functools
import uuid

import pyro
import pyro.infer
import pyro.contrib.named as named


def _lift(fn, _namespace=None, _latent=None):
    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        latent = named.Object(_namespace) if _latent is None else _latent
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
            lifted_data = site_collector.visit(lambda name, val, acc: acc.setdefault(name, val),
                                               {})
        pyro.condition(_lift(fn, _latent=latent),
                       data=lifted_data)(*args, **kwargs)
    return partial_condition


@functools.wraps(pyro.infer.SVI)
def SVI(model, guide, *args, **kwargs):
    namespace = str(uuid.uuid4())
    obj = pyro.infer.SVI(_lift(model, namespace),
                         _lift(guide, namespace), *args, **kwargs)
    obj._namespace = namespace
    return obj


@functools.wraps(pyro.infer.Importance)
def Importance(model, guide=None, *args, **kwargs):
    namespace = str(uuid.uuid4())
    obj = pyro.infer.Importance(_lift(model, namespace),
                                _lift(guide, namespace) if guide is not None else None,
                                *args, **kwargs)
    obj._namespace = namespace
    return obj


@functools.wraps(pyro.infer.Search)
def Search(model, *args, **kwargs):
    namespace = str(uuid.uuid4())
    obj = pyro.infer.Search(_lift(model, namespace),
                            *args, **kwargs)
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
        sites = list(site_collector.visit(lambda name, val, acc: acc.add(name),
                                          set()))
    marginal = pyro.infer.Marginal(trace_dist, sites, *args, **kwargs)

    # Intercept the call function for marginal.
    def call(*args, **kwargs):
        marginal_out = marginal(*args, **kwargs)
        # Rewrite returned marginals to be an object.
        if isinstance(marginal_out, dict):
            ret = {}
            for key, value in marginal_out.items():
                ret[key[len(namespace + '.'):]] = value
            return ret
        else:
            return marginal_out
    return call
