from torch.autograd import Variable as V
from torch import Tensor as T
from torch import is_tensor, cat
from sys import float_info

# ----------------------------------------------------------------------------
# Globals / constants


# create torch variable around tesnor with input X
def A(x): return [x]


def TA(x): return T(A(x))


def VT(x): return V(T(x))


def VTA(x): return V(TA(x))


# basic vector additions and lerping
class Vector2():
    def __init__(self, x=0, y=0):
        if is_tensor(x) or isinstance(x, V):
            x, y = x.data[0], y.data[0]

        self.pt_val = VT([x, y])

    @property
    def x(self):
        return self.pt_val.data[0]

    @property
    def y(self):
        return self.pt_val.data[1]

    def __repr__(self):
        return "{:0.5},{:0.5}".format(self.x, self.y)

    def add(self, other):
        self.pt_val += other.pt_val
        return self

    def clone(self):
        return Vector2(*self.pt_val.data)

    # linear interpolate between two points us and them
    def lerp(self, other, zero_one):
        self.pt_val = zero_one*self.pt_val + (1 - zero_one)*other.pt_val
        return self

    # obs multiply x,y by scalar
    def scalar_mult(self, mult):
        self.pt_val *= mult
        return self

    def concat_vals(self, other):
        return cat([self.pt_val.unsqueeze(0), other.pt_val.unsqueeze(0)], 0)

    def min(self, other):
        mv, _ = self.concat_vals(other).min(1)
        self.pt_val = mv
        return self

    def max(self, other):
        mv, _ = self.concat_vals(other).max(1)
        self.pt_val = mv
        return self


class BBOX2():
    def __init__(self, v2_min=None, v2_max=None):
        self.min = v2_min if v2_min is not None else Vector2(float_info.max, float_info.max)
        self.max = v2_max if v2_max is not None else Vector2(-float_info.max, -float_info.max)

    def clone(self):
        return BBOX2(self.min.clone(), self.max.clone())

    def expand_by_point(self, point):
        self.min.min(point)
        self.max.max(point)

    # overwrites bbox with combined union -- clone to avoid issues
    def union(self, other):
        self.min.min(other.min)
        self.max.max(other.max)
        return self


class Viewport():

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def clone(self):
        return Viewport(self.xmin, self.xmax, self.ymin, self.ymax)
