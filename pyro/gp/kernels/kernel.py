from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

from six import add_metaclass
import torch.nn as nn


@add_metaclass(ABCMeta)
class Kernel(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()
        pass



    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)


class Add(Kernel):
    pass
