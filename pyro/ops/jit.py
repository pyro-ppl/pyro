import weakref
import torch

import pyro
import pyro.poutine as poutine


class CompiledFunction(object):
    """
    Output type of :func:`pyro.ops.jit.compile`.

    Wrapper around the output of :func:`torch.jit.compile`
    that handles parameter plumbing.

    The actual PyTorch compilation artifact is stored in :attr:`compiled`.
    Call diagnostic methods on this attribute.
    """
    def __init__(self, fn, **jit_options):
        self.fn = fn
        self._jit_options = jit_options
        self.compiled = None
        self._param_names = None

    def __call__(self, *args, **kwargs):

        # if first time
        if self.compiled is None:
            # param capture
            with poutine.block():
                with poutine.trace(param_only=True) as first_param_capture:
                    self.fn(*args, **kwargs)

            self._param_names = list(set(first_param_capture.trace.nodes.keys()))

            weakself = weakref.ref(self)

            @torch.jit.compile(**self._jit_options)
            def compiled(unconstrained_params, *args):
                self = weakself()
                constrained_params = {}
                for name, unconstrained_param in zip(self._param_names, unconstrained_params):
                    constrained_param = pyro.param(name)  # assume param has been initialized
                    assert constrained_param.unconstrained() is unconstrained_param
                    constrained_params[name] = constrained_param

                return poutine.replay(
                    self.fn, params=constrained_params)(*args, **kwargs)

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


def compile(fn=None, **jit_options):
    """
    Drop-in replacement for :func:`torch.jit.compile` that works with
    Pyro functions that call :func:`pyro.param`.

    The actual compilation artifact is stored in the ``compiled`` attribute of the output.
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
    if fn is None:
        return lambda fn: compile(fn, **jit_options)
    return CompiledFunction(fn, **jit_options)
