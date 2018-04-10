from __future__ import absolute_import, division, print_function

import functools
from six.moves import xrange

from pyro.poutine import util

from .block_poutine import BlockMessenger
from .condition_poutine import ConditionMessenger
from .enumerate_poutine import EnumerateMessenger  # noqa: F401
from .escape_poutine import EscapeMessenger
from .indep_poutine import IndepMessenger  # noqa: F401
from .infer_config_poutine import InferConfigMessenger
from .lift_poutine import LiftMessenger
from .poutine import _PYRO_STACK, Messenger  # noqa: F401
from .replay_poutine import ReplayMessenger
from .scale_poutine import ScaleMessenger
from .trace import Trace  # noqa: F401
from .trace_poutine import TraceMessenger


block = BlockMessenger
condition = ConditionMessenger
enum = EnumerateMessenger
escape = EscapeMessenger
indep = IndepMessenger
infer_config = InferConfigMessenger
lift = LiftMessenger
replay = ReplayMessenger
scale = ScaleMessenger
trace = TraceMessenger


def do(data):

    # @do(data)
    # def fn(...):
    #     ...

    def _fn(fn):
        return block(hide=list(data.keys()))(
            condition(data=data)(fn))

    return _fn


def queue(queue, max_tries=None,
          extend_fn=None, escape_fn=None, num_samples=None):
    """
    :param queue: a queue data structure like multiprocessing.Queue to hold partial traces
    :param max_tries: maximum number of attempts to compute a single complete trace
    :param extend_fn: function (possibly stochastic) that takes a partial trace and a site
    and returns a list of extended traces
    :param escape_fn: function (possibly stochastic) that takes a partial trace and a site
    and returns a boolean value to decide whether to exit
    :param num_samples: optional number of extended traces for extend_fn to return
    :returns: stochastic function wrapped in poutine logic

    Given a queue,
    return wrapper that will get a return value from a complete trace in the queue
    """

    if max_tries is None:
        max_tries = int(1e6)

    if extend_fn is None:
        extend_fn = util.enum_extend

    if escape_fn is None:
        escape_fn = util.discrete_escape

    if num_samples is None:
        num_samples = -1

    def queue_decorator(fn):
        def _fn(*args, **kwargs):

            for i in xrange(max_tries):
                assert not queue.empty(), \
                    "trying to get() from an empty queue will deadlock"

                next_trace = queue.get()
                try:
                    ftr = trace(escape(replay(fn, next_trace),
                                       functools.partial(escape_fn, next_trace)))
                    return ftr(*args, **kwargs)
                except util.NonlocalExit as site_container:
                    site_container.reset_stack()
                    for tr in extend_fn(ftr.trace.copy(), site_container.site,
                                        num_samples=num_samples):
                        queue.put(tr)

            raise ValueError("max tries ({}) exceeded".format(str(max_tries)))
        return _fn

    return queue_decorator
