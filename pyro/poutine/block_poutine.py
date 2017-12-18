from __future__ import absolute_import, division, print_function

from .poutine import Poutine


class BlockPoutine(Poutine):
    """
    This Poutine selectively hides pyro primitive sites from the outside world.

    For example, suppose the stochastic function fn has two sample sites "a" and "b".
    Then any poutine outside of BlockPoutine(fn, hide=["a"])
    will not be applied to site "a" and will only see site "b":

    >>> fn_inner = TracePoutine(fn)
    >>> fn_outer = TracePoutine(BlockPoutine(TracePoutine(fn), hide=["a"]))
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

    BlockPoutine has a flexible interface that allows users
    to specify in several different ways
    which sites should be hidden or exposed.
    See the constructor for details.
    """

    def __init__(self, fn,
                 hide_all=True, expose_all=False,
                 hide=None, expose=None,
                 hide_types=None, expose_types=None):
        """
        :param bool hide_all: hide all sites
        :param bool expose_all: expose all sites normally
        :param list hide: list of site names to hide, rest will be exposed normally
        :param list expose: list of site names to expose, rest will be hidden
        :param list hide_types: list of site types to hide, rest will be exposed normally
        :param list expose_types: list of site types to expose normally, rest will be hidden

        Constructor for blocking poutine
        Default behavior: block everything (hide_all == True)

        A site is hidden if at least one of the following holds:
        1. msg["name"] in hide
        2. msg["type"] in hide_types
        3. msg["name"] not in expose and msg["type"] not in expose_types
        4. hide_all == True
        """
        super(BlockPoutine, self).__init__(fn)
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
        assert set(hide).isdisjoint(set(expose)), \
            "cannot hide and expose a site"

        # hide_types and expose_types intersect?
        if hide_types is None:
            hide_types = []
        if expose_types is None:
            expose_types = []
        assert set(hide_types).isdisjoint(set(expose_types)), \
            "cannot hide and expose a site type"

        # now set stuff
        self.hide_all = hide_all
        self.expose_all = expose_all
        self.hide = hide
        self.expose = expose
        self.hide_types = hide_types
        self.expose_types = expose_types

    def _block_up(self, msg):
        """
        :param msg: current message at a trace site, after all execution finished.
        :returns: boolean decision to hide or expose site.

        A site is hidden if at least one of the following holds:
        1. msg["name"] in self.hide
        2. msg["type"] in self.hide_types
        3. msg["name"] not in self.expose and msg["type"] not in self.expose_types
        4. self.hide_all == True
        """
        # handle observes
        if msg["type"] == "sample" and msg["is_observed"]:
            msg_type = "observe"
        else:
            msg_type = msg["type"]

        # decision rule for hiding:
        if (msg["name"] in self.hide) or \
           (msg_type in self.hide_types) or \
           ((msg["name"] not in self.expose) and
            (msg_type not in self.expose_types) and self.hide_all):  # noqa: E129

            return True
        # otherwise expose
        else:
            return False

    def _pyro_sample(self, msg):
        """
        :param msg:  current message at a trace site
        :returns: a sample from the stochastic function at the site.

        Default sample behavior with a side effect.
        Applies self._block_up to decide whether to hide the site.
        """
        ret = super(BlockPoutine, self)._pyro_sample(msg)
        msg["stop"] = self._block_up(msg)
        return ret

    def _pyro_param(self, msg):
        """
        :param msg:  current message at a trace site
        :returns: the result of querying the parameter store.

        Default param behavior with a side effect.
        Applies self._block_up to decide whether to hide the site.
        """
        ret = super(BlockPoutine, self)._pyro_param(msg)
        msg["stop"] = self._block_up(msg)
        return ret
