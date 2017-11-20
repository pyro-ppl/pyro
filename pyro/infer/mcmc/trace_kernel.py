from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta

from six import add_metaclass


@add_metaclass(ABCMeta)
class TraceKernel(object):
    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def num_accepts(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, trace, *args, **kwargs):
        raise NotImplementedError
