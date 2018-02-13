from __future__ import absolute_import, division, print_function

from .poutine import _PYRO_STACK


def site_is_subsample(site):
    """
    Determines whether a trace site originated from a subsample statement inside an `iarange`.
    """
    return site["type"] == "sample" and type(site["fn"]).__name__ == "_Subsample"


def prune_subsample_sites(trace):
    """
    Copies and removes all subsample sites from a trace.
    """
    trace = trace.copy()
    for name, site in list(trace.nodes.items()):
        if site_is_subsample(site):
            trace.remove_node(name)
    return trace


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

    def reset_stack(self):
        """
        Reset the state of the frames remaining in the stack.
        Necessary for multiple re-executions in poutine.queue.
        """
        for frame in _PYRO_STACK:
            frame._reset()
