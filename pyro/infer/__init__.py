from pyro.infer.abstract_infer import AbstractInfer
from pyro.infer.mh import MH
from pyro.infer.importance import Importance
from pyro.infer.search import Search

from .kl_qp import KL_QP

def marginal(trace_posterior):
    raise NotImplementedError("marginal not implemented yet")
