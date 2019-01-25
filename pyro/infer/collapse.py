from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import pyro.ops.packed as packed
import pyro.poutine as poutine
from pyro.infer.discrete import infer_discrete
from pyro.ops.contract import contract_tensor_tree
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import MapRing, SampleRing


def _is_collapsed(node):
    return node["type"] == "sample" and node["infer"].get("enumerate") == "parallel"


def collapse(model, first_available_dim):
    """
    A handler to collapse sample sites marked with
    ``site["infer"]["collapse"] = True``.
    Collapsed sites will be blocked.

    .. warning:: Cannot be wrapped with :func:~`pyro.poutine.replay`

    :param model: a stochastic function (callable containing Pyro primitive calls)
    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer.
    """
    collapsed_model = infer_discrete(model, first_available_dim, temperature=1)
    return poutine.block(collapsed_model, hide_fn=_is_collapsed)
