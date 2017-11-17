from torch.autograd import Variable as V
from torch import Tensor as T

# ----------------------------------------------------------------------------
# Globals / constants


# create torch variable around tesnor with input X
def A(x): return [x]


def TA(x): return T(A(x))


def VT(x): return V(T(x))


def VTA(x): return V(TA(x))
