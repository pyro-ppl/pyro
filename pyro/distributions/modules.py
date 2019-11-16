import pyro.distributions as dist
from pyro.distributions.module import dist_as_module

__all__ = []

for name in dist.__all__:
    cls = getattr(dist, name)
    if isinstance(cls, type):
        if issubclass(cls, dist.TorchDistribution):
            if cls is not dist.TorchDistribution:
                locals()[name] = dist_as_module(cls)
                __all__.append(name)

del name, cls, dist_as_module
