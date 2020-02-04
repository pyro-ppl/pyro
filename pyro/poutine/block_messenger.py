# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from pyro.poutine.messenger import Messenger


def _block_fn(expose, expose_types, hide, hide_types, hide_all, msg):
    # handle observes
    if msg["type"] == "sample" and msg["is_observed"]:
        msg_type = "observe"
    else:
        msg_type = msg["type"]

    is_not_exposed = (msg["name"] not in expose) and \
                     (msg_type not in expose_types)

    # decision rule for hiding:
    if (msg["name"] in hide) or \
            (msg_type in hide_types) or \
            (is_not_exposed and hide_all):  # noqa: E129

        return True
    # otherwise expose
    else:
        return False


def _make_default_hide_fn(hide_all, expose_all, hide, expose, hide_types, expose_types):
    # first, some sanity checks:
    # hide_all and expose_all intersect?
    assert (hide_all is False and expose_all is False) or \
        (hide_all != expose_all), "cannot hide and expose a site"

    # hide and expose intersect?
    if hide is None:
        hide = []
    else:
        hide_all = False

    if expose is None:
        expose = []
    else:
        hide_all = True

    assert set(hide).isdisjoint(set(expose)), \
        "cannot hide and expose a site"

    # hide_types and expose_types intersect?
    if hide_types is None:
        hide_types = []
    else:
        hide_all = False

    if expose_types is None:
        expose_types = []
    else:
        hide_all = True

    assert set(hide_types).isdisjoint(set(expose_types)), \
        "cannot hide and expose a site type"

    return partial(_block_fn, expose, expose_types, hide, hide_types, hide_all)


class BlockMessenger(Messenger):
    """
    This handler selectively hides Pyro primitive sites from the outside world.
    Default behavior: block everything.

    A site is hidden if at least one of the following holds:

        0. ``hide_fn(msg) is True`` or ``(not expose_fn(msg)) is True``
        1. ``msg["name"] in hide``
        2. ``msg["type"] in hide_types``
        3. ``msg["name"] not in expose and msg["type"] not in expose_types``
        4. ``hide``, ``hide_types``, and ``expose_types`` are all ``None``

    For example, suppose the stochastic function fn has two sample sites "a" and "b".
    Then any effect outside of ``BlockMessenger(fn, hide=["a"])``
    will not be applied to site "a" and will only see site "b":

        >>> def fn():
        ...     a = pyro.sample("a", dist.Normal(0., 1.))
        ...     return pyro.sample("b", dist.Normal(a, 1.))
        >>> fn_inner = pyro.poutine.trace(fn)
        >>> fn_outer = pyro.poutine.trace(pyro.poutine.block(fn_inner, hide=["a"]))
        >>> trace_inner = fn_inner.get_trace()
        >>> trace_outer  = fn_outer.get_trace()
        >>> "a" in trace_inner
        True
        >>> "a" in trace_outer
        False
        >>> "b" in trace_inner
        True
        >>> "b" in trace_outer
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param hide_fn: function that takes a site and returns True to hide the site
        or False/None to expose it.  If specified, all other parameters are ignored.
        Only specify one of hide_fn or expose_fn, not both.
    :param expose_fn: function that takes a site and returns True to expose the site
        or False/None to hide it.  If specified, all other parameters are ignored.
        Only specify one of hide_fn or expose_fn, not both.
    :param bool hide_all: hide all sites
    :param bool expose_all: expose all sites normally
    :param list hide: list of site names to hide
    :param list expose: list of site names to be exposed while all others hidden
    :param list hide_types: list of site types to be hidden
    :param lits expose_types: list of site types to be exposed while all others hidden
    :returns: stochastic function decorated with a :class:`~pyro.poutine.block_messenger.BlockMessenger`
    """

    def __init__(self, hide_fn=None, expose_fn=None,
                 hide_all=True, expose_all=False,
                 hide=None, expose=None,
                 hide_types=None, expose_types=None):
        super().__init__()
        if not (hide_fn is None or expose_fn is None):
            raise ValueError("Only specify one of hide_fn or expose_fn")
        if hide_fn is not None:
            self.hide_fn = hide_fn
        elif expose_fn is not None:
            self.hide_fn = lambda msg: not expose_fn(msg)
        else:
            self.hide_fn = _make_default_hide_fn(hide_all, expose_all,
                                                 hide, expose,
                                                 hide_types, expose_types)

    def _process_message(self, msg):
        msg["stop"] = bool(self.hide_fn(msg))
        return None
