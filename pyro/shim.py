from __future__ import absolute_import, division, print_function

import contextlib
import re

import torch


def parse_torch_version():
    """
    Parses `torch.__version__` into a semver-ish version tuple.
    This is needed to handle subpatch `_n` parts outside of the semver spec.

    :returns: a tuple `(major, minor, patch, extra_stuff)`
    """
    match = re.match(r"(\d\.\d\.\d)(.*)", torch.__version__)
    major, minor, patch = map(int, match.group(1).split("."))
    extra_stuff = match.group(2)
    return major, minor, patch, extra_stuff


# Polyfill to bridge the change of .volatile between PyTorch 0.3 and 0.4.
try:
    # These work in PyTorch 0.4 prerelease.
    torch_no_grad = torch.no_grad

    def is_volatile(variable):
        return False

except AttributeError:
    # These work in PyTorch 0.3 and earlier.

    @contextlib.contextmanager
    def torch_no_grad():
        yield

    def is_volatile(variable):
        return variable.volatile
