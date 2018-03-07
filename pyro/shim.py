from __future__ import absolute_import, division, print_function

import re

import torch


def parse_torch_version():
    """
    Parses ``torch.__version__`` into a semver-ish version tuple.
    This is needed to handle subpatch `_n` parts outside of the semver spec.

    :returns: a tuple `(major, minor, patch, extra_stuff)`
    """
    match = re.match(r"(\d\.\d\.\d)(.*)", torch.__version__)
    major, minor, patch = map(int, match.group(1).split("."))
    extra_stuff = match.group(2)
    return major, minor, patch, extra_stuff
