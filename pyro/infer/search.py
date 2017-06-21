import pyro
from pyro.infer.poutine import Poutine, TagPoutine
import torch
from torch.autograd import Variable
# from pyro.distributions import Discrete
# from collections import OrderedDict


class Trace(dict):

    def __init__(self, *args):
        super(Trace, self).__init__(*args)
        self.log_pdf = 0.0
        # TODO: put in infrastructure to fold over trace as it's built.
        #   that speeds up running sums and such.

    def add_sample(self, name, sample, logpdf):
        node = {}
        # TODO: make this as an object instead of dict?
        node["type"] = "sample"
        node["sample"] = sample
        node["log_pdf"] = logpdf
        self.log_pdf += logpdf
        self[name] = node

    def add_observe(self, name, logpdf):
        node = {}
        node["type"] = "observe"
        node["log_pdf"] = logpdf
        self.log_pdf += logpdf
        self[name] = node

    def copy(self):
        # will this work?
        return Trace(self)


class SearchCo(Poutine):
    """
    This coroutine:
        at sample: extends the current trace with all elements of the
        support and non-local exit with extended traces.
        at observe: extends trace with observation and keeps going.

    TODO!!
    At sample we yield non-locally from the underlying function
    back to the queue handler. Currently this is done by raise an exception.
    This is sound but ugly. Is there a saner control flow method in python?

    Should be initialized with teh underlying fn and a queue object.
    """

    def set_trace(self, trace):
        self.trace = trace

    def __init__(self, *args, **kwargs):
        super(SearchCo, self).__init__(*args, **kwargs)
        self.trace = None

    # def _enter_poutine(self, *args, **kwargs):
    #     """
    #     When model execution begins
    #     """
    #     super(SearchCo, self)._enter_poutine(*args, **kwargs)

    def _pyro_sample(self, name, dist):

        # if we've sampled this previously re-use value.
        # TODO: implement via poutine layer?
        if name in self.trace:
            return self.trace[name]["sample"]

        support = dist.support()

        def extend_trace(s):
            # FIXME: use tree in memory rather than copy?
            trace = self.trace.copy()
            trace.add_sample(name, s, dist.log_pdf(s))
            return trace

        traces = map(extend_trace, support)

        raise ReturnExtendedTraces(traces)

    def _pyro_observe(self, name, dist, val):
        self.trace.add_observe(name, dist.log_pdf(val))
        return val

    # TODO: leverage independence contract of map_data.


class ReturnExtendedTraces(Exception):
    def __init__(self, traces, *args, **kwargs):
        super(ReturnExtendedTraces, self).__init__(*args, **kwargs)
        self.traces = traces


class Search(object):
    """
    The main search-based inference class.
    Constructs a trace stream and then consumes it into a histogram-based
    marginal distribution.

    This should be initialized with a queue object that embodies the
    desired search policy. queue should implement:
        queue.put(thing)
        queue.get()
    """

    def __init__(self, model, queue=None, *args, **kwargs):
        super(Search, self).__init__(*args, **kwargs)
        self.model = SearchCo(model)
        if not queue:
            # default queue
            self.queue = FILOQ()
        else:
            self.queue = queue

    def step(self, *args, **kwargs):
        # this defines an iteration over complete traces.
        # TODO: bail out after fixed number of steps?
        while not self.queue.empty():
            try:
                next_trace = self.queue.get()
                self.model.set_trace(next_trace)
                rv = self.model(*args, **kwargs)
            except ReturnExtendedTraces as returned_traces:
                # non-local exit from sample call.
                # push onto queue.
                for t in returned_traces.traces:
                    self.queue.put(t)
            else:
                # model returned without hitting another sample.
                # yield return value and score to whoever is consuming us.
                yield rv, self.model.trace.log_pdf

    def __call__(self, *args, **kwargs):
        """
        main method: accumulate complete traces into marginal histogram.
        """

        # initialize the queue with empty trace
        self.queue.put(Trace())

        # return generator object for completed executions
        return self.step(*args, **kwargs)


class Marginal(object):
    # NOT tested
    def __init__(self, posterior, *args, **kwargs):
        super(Marginal, self).__init__(*args, **kwargs)
        # TODO handle other kinds of posterior objects...
        self.posterior = posterior
        self.cache = {}  # better cache data structure?

    def __call__(self, *args, **kwargs):
        # check args against cache
        key = [args, kwargs]  # stringify?
        if key in self.cache:
            return self.cache[key]
        else:
            # TODO: use itertools for speed?
            hist = {}
            for rv, logpdf in self.posterior(*args, **kwargs):
                # TODO: what can be keys in python? need to stringify rv?
                if rv not in hist:
                    hist[rv] = 0.0
                hist[rv] += logpdf

            # TODO: normalize
            # FIXME: make discrete dist.
            dist = hist  # Discrete(hist.keys(), hist.vals())
            return dist


class FILOQ(list):

    def put(self, x):
        self.append(x)

    def get(self):
        return self.pop()

    def empty(self):
        return len(self) == 0
