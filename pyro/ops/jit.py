import weakref
# import functools
import torch

import pyro
import pyro.poutine as poutine


class compile(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._compiled = None
        self._param_names = None

    def __call__(self, fn):

        # @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # if first time
            if self._compiled is None:
                # param capture
                with poutine.block():
                    with poutine.trace(param_only=True) as first_param_capture:
                        fn(*args, **kwargs)

                self._param_names = list(set(first_param_capture.trace.nodes.keys()))

                weakself = weakref.ref(self)

                @torch.jit.compile(**self._kwargs)
                def compiled(unconstrained_params, *args):
                    self = weakself()
                    constrained_params = {name: pyro.param(name, unconstrained_param)
                                          for name, unconstrained_param
                                          in zip(self._param_names,
                                                 unconstrained_params)}

                    return poutine.replay(
                        fn, params=constrained_params)(*args, **kwargs)

                self._compiled = compiled

            param_list = [pyro.param(name).unconstrained()
                          for name in self._param_names]

            with poutine.block(hide=self._param_names):
                with poutine.trace(param_only=True) as param_capture:
                    ret = self._compiled(param_list, *args, **kwargs)

                    new_params = filter(lambda name: name not in self._param_names,
                                        param_capture.trace.nodes.keys())

            for name in new_params:
                # enforce uniqueness
                if name not in self._param_names:
                    self._param_names.append(name)

            return ret

        return wrapper
