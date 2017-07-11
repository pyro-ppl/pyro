import pyro
import torch
import sys
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

from .poutine import Poutine


class ReturnExtendedTraces(Exception):
    def __init__(self, traces, *args, **kwargs):
        super(ReturnExtendedTraces, self).__init__(*args, **kwargs)
        self.traces = traces


class QueuePoutine(Poutine):
    """
    Poutine for enumerating a queue of traces
    Useful for systematic search, beam search
    """
    def __init__(self, fn, queue=None, max_tries=None):
        """
        Constructor.
        All persistent state is in the queue.
        """
        super(QueuePoutine, self).__init__(fn)
        self.transparent = False
        if queue is None:
            queue = Queue()
            queue.put(pyro.poutine.Trace())
        self.queue = queue
        if max_tries is None:
            max_tries = 1e6
        self.max_tries = int(max_tries)

    def __call__(self, *args, **kwargs):
        """
        Keep going until it returns a completed trace from the queue
        or has run for too long
        """
        for i in range(self.max_tries):
            try:
                ret_val = super(QueuePoutine, self).__call__(*args, **kwargs)
                return ret_val
            except ReturnExtendedTraces as returned_traces:
                self._exit_poutine(None, *args, **kwargs)
                self._flush_stack()
                for tr in returned_traces.traces:
                    self.queue.put(tr)
        raise ValueError("max tries ({}) exceeded".format(str(self.max_tries)))

    def _enter_poutine(self, *args, **kwargs):
        """
        Set a guide trace and a pivot switch
        """
        self.pivot_seen = False
        self.guide_trace = self.queue.get()

    def _exit_poutine(self, *args, **kwargs):
        """
        Forget the guide and pivot switch
        """
        self.pivot_seen = False
        self.guide_trace = None  # XXX what to put here?

    def _pyro_sample(self, prev_val, name, fn, *args, **kwargs):
        """
        Return the sample in the guide trace when appropriate
        """
        assert hasattr(fn, "support"), "distribution has no support method"
        if name in self.guide_trace:
            assert self.guide_trace[name]["type"] == "sample", \
                "site {} in guide_trace is not a sample".format(name)
            return self.guide_trace[name]["value"]
        elif not self.pivot_seen:
            self.pivot_seen = True
            extended_traces = []
            for s in fn.support():
                extended_traces.append(
                    self.guide_trace.copy().add_sample(name, s, fn, *args, **kwargs))
            raise ReturnExtendedTraces(extended_traces)
        else:
            raise ValueError("should never get here (malfunction at site {})".format(name))
