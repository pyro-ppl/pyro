from __future__ import absolute_import, division, print_function


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
