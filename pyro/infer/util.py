from __future__ import absolute_import, division, print_function

import numbers
import warnings

import numpy as np

import torch


def _warn_if_nan(name, variable):
    value = variable.data[0]
    if np.isnan(value):
        warnings.warn("Encountered NAN log_pdf at site '{}'".format(name))
    if np.isinf(value) and value > 0:
        warnings.warn("Encountered +inf log_pdf at site '{}'".format(name))
    # Note that -inf log_pdf is fine: it is merely a zero-probability event.


def torch_data_sum(x):
    """
    Like ``x.data.sum()`` for a ``torch.autograd.Variable``, but also works
    with numbers.
    """
    if isinstance(x, numbers.Number):
        return x
    return x.data.sum()


def torch_sum(x):
    """
    Like ``x.sum()`` for a ``torch.autograd.Variable``, but also works with
    numbers.
    """
    if isinstance(x, numbers.Number):
        return x
    return x.sum()


def torch_backward(x):
    """
    Like ``x.backward()`` for a ``torch.autograd.Variable``, but also accepts
    numbers (a no-op if given a number).
    """
    if isinstance(x, torch.autograd.Variable):
        x.backward()


def trace_log_pdf(trace, site_filter=lambda name, site: True):
    """
    Compute the local and overall log-probabilities of the trace.

    The local computation is memoized.

    :returns: total log probability.
    :rtype: torch.autograd.Variable
    """
    log_p = 0.0
    for name, site in trace.nodes.items():
        if site["type"] == "sample" and site_filter(name, site):
            try:
                site_log_p = site["log_pdf"]
            except KeyError:
                args, kwargs = site["args"], site["kwargs"]
                site_log_p = site["fn"].log_pdf(
                    site["value"], *args, **kwargs) * site["scale"]
                site["log_pdf"] = site_log_p
                _warn_if_nan(name, site_log_p)
            log_p += site_log_p
    return log_p


# XXX This only makes sense when all tensors have compatible shape.
def trace_batch_log_pdf(trace, site_filter=lambda name, site: True, sum_sites=True):
    """
    Compute the batched local and overall log-probabilities of the trace.

    The local computation is memoized, and also stores the local `.log_pdf()`.
    """
    log_p = 0.0
    for name, site in trace.nodes.items():
        if site["type"] == "sample" and site_filter(name, site):
            try:
                site_log_p = site["batch_log_pdf"]
            except KeyError:
                args, kwargs = site["args"], site["kwargs"]
                site_log_p = site["fn"].batch_log_pdf(
                    site["value"], *args, **kwargs) * site["scale"]
                site["batch_log_pdf"] = site_log_p
                site["log_pdf"] = site_log_p.sum()
                _warn_if_nan(name, site["log_pdf"])
            if sum_sites:
                # Here log_p may be broadcast to a larger tensor:
                log_p = log_p + site_log_p
    return log_p
