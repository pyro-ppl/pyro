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
                for tr in returned_traces.traces:
                    self.queue.put(tr)
        raise ValueError("max tries ({}) exceeded".format(str(self.max_tries)))

    def __enter__(self):
        """
        Set a guide trace and a pivot switch
        """
        self.pivot_seen = False
        self.guide_trace = self.queue.get()
        return super(QueuePoutine, self).__enter__()

    def __exit__(self, *args):
        """
        Forget the guide and pivot switch
        """
        self.pivot_seen = False
        self.guide_trace = None  # XXX what to put here?
        return super(QueuePoutine, self).__exit__(*args)

    def _pyro_sample(self, msg):
        """
        Samples continuous variables and enumerates discrete variables.

        Discrete variables are those that implement a `.support()` method.
        Discrete variables are enumerated by raising-and-replaying.

        :returns: A sample.
        :raises: ReturnExtendedTraces.
        """
        name, fn, args, kwargs = \
            msg["name"], msg["fn"], msg["args"], msg["kwargs"]
        assert hasattr(fn, "support"), "distribution has no support method"
        if name in self.guide_trace:
            assert self.guide_trace[name]["type"] == "sample", \
                "site {} in guide_trace is not a sample".format(name)
            msg["done"] = True
            return self.guide_trace[name]["value"]

        try:
            support = fn.support(*args, **kwargs)
        except (AttributeError, NotImplementedError):
            # For distributions without discrete support, we sample as usual.
            r_val = super(QueuePoutine, self)._pyro_sample(msg)
            self.guide_trace.add_sample(name, msg["scale"], r_val, fn, *args, **kwargs)
            return r_val

        assert not self.pivot_seen, "should never get here (malfunction at site {})".format(name)
        self.pivot_seen = True
        extended_traces = [
            self.guide_trace.copy().add_sample(name, msg["scale"], s, fn, *args, **kwargs)
            for s in support
        ]
        msg["done"] = True
        raise ReturnExtendedTraces(extended_traces)
