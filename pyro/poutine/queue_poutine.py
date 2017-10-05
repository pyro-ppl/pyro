from six.moves.queue import Queue

import pyro
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

    def _exit_poutine(self, r_val, *args, **kwargs):
        """
        Forget the guide and pivot switch
        """
        self.pivot_seen = False
        self.guide_trace = None  # XXX what to put here?
        return r_val

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Samples continuous variables and enumerates discrete variables.

        Discrete variables are those that implement a `.support()` method.
        Discrete variables are enumerated by raising-and-replaying.

        :returns: A sample.
        :raises: ReturnExtendedTraces.
        """
        if name in self.guide_trace:
            assert self.guide_trace[name]["type"] == "sample", \
                "site {} in guide_trace is not a sample".format(name)
            msg["done"] = True
            return self.guide_trace[name]["value"]
        assert not self.pivot_seen, "should never get here (malfunction at site {})".format(name)
        self.pivot_seen = True

        try:
            support = fn.support(*args, **kwargs)
        except (AttributeError, NotImplementedError):
            # For distributions without discrete support, we sample as usual.
            return super(QueuePoutine, self)._pyro_sample(self, msg, name, fn, *args, **kwargs)
        extended_traces = [
            self.guide_trace.copy().add_sample(name, msg["scale"], s, fn, *args, **kwargs)
            for s in support
        ]
        msg["done"] = True
        raise ReturnExtendedTraces(extended_traces)
