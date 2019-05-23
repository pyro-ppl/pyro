from __future__ import absolute_import, division, print_function

import re
from abc import ABCMeta, abstractmethod

from contextlib2 import ExitStack
from six import add_metaclass

from pyro.contrib.autoguide import AutoGuide


def _numel(shape):
    """
    Computes the total number of elements of a given tensor shape.
    """
    result = 1
    for size in shape:
        result *= size
    return result


@add_metaclass(ABCMeta)
class EasyGuide(AutoGuide):
    def __init__(model, prefix="easy"):
        super(EasyGuide, self).__init__(model, prefix=prefix)
        self._sample_meta = {}

    @abstractmethod
    def guide(self, *args, **kargs):
        raise NotImplementedError

    def __call__(self, *args, **kargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)
        self.plates = self._create_plates()
        return self.guide(*args, **kwargs)

    def sample(self, guide_name, easy_fn, match=".*"):
        if guide_name not in self._sample_meta:
            meta = self._sample_meta[guide_name]
        else:
            meta = {"model_sites": []}
            for model_name, site in self.prototype_trace.iter_stochastic_nodes():
                if re.match(match, model_name):
                    meta["model_sites"].append(site)
            if not meta["model_sites"]:
                raise ValueError("At EasyGuide site {}, pattern {} matched no model sites"
                                 .format(repr(guide_name), repr(match)))
            self._sample_meta[guide_name] = meta

        for site in meta["model_sites"]:
            pass

        guide_z = pyro.sample(guide_name, easy_fn(sites),
                              infer={"is_auxiliary": True})

        model_zs = {}
        pos = 0
        for site in sites:
            size = _numel(fn.shape)
            unconstrained_z = guide_z[..., pos: pos + size]
            pos += size

            constrained_z = transform_to(fn.support)(unconstrained_z)
            transform = biject_to(site["fn"].support)
            z = transform(unconstrained_z)
            log_density = transform.inv.log_abs_det_jacobian(z, unconstrained_z)
            log_density = sum_rightmost(log_density, log_density.dim() - z.dim() + site["fn"].event_dim)
            delta_dist = dist.Delta(z, log_density=log_density, event_dim=site["fn"].event_dim)

            model_name = site["name"]
            with ExitStack() as stack:
                for frame in self._cond_indep_stacks[model_name]:
                    stack.enter_context(plates[frame.name])
                model_zs[model_name] = pyro.sample(name, delta_dist)

        return guide_z, model_zs

    def param(self, guide_name, init_fn):
        pass
