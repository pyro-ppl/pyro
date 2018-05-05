import weakref
import torch

import pyro
import pyro.poutine as poutine


class compile(object):
    """
    Drop-in replacement for :func:`torch.jit.compile` that works with
    Pyro functions that call :func:`pyro.param`.

    The actual compilation artifact is stored in ``self.compiled``.
    Call diagnostic methods on this attribute.

    Example::

        def model(x):
            scale = pyro.param("scale", torch.tensor(0.5), constraint=constraints.positive)
            return pyro.sample("y", dist.Normal(x, scale))

        @pyro.ops.jit.compile(nderivs=1)
        def model_log_prob_fn(x, y):
            cond_model = pyro.condition(model, data={"y": y})
            tr = pyro.poutine.trace(cond_model).get_trace(x)
            return tr.log_prob_sum()
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.compiled = None
        self._param_names = None

    def __call__(self, fn):

        assert self.compiled is None, \
            "Cannot use a compile instance on more than one function"

        def wrapper(*args, **kwargs):
            # if first time
            if self.compiled is None:
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

                self.compiled = compiled

            param_list = [pyro.param(name).unconstrained()
                          for name in self._param_names]

            with poutine.block(hide=self._param_names):
                with poutine.trace(param_only=True) as param_capture:
                    ret = self.compiled(param_list, *args, **kwargs)

            new_params = filter(lambda name: name not in self._param_names,
                                param_capture.trace.nodes.keys())

            for name in new_params:
                # enforce uniqueness
                if name not in self._param_names:
                    self._param_names.append(name)

            return ret

        return wrapper
