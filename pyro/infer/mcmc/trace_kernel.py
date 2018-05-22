from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

from six import add_metaclass


@add_metaclass(ABCMeta)
class TraceKernel(object):

    def setup(self, *args, **kwargs):
        """
        Optional method to set up any state required at the start of the
        simulation run.

        :param \*args: Algorithm specific positional arguments.
        :param \*\*kwargs: Algorithm specific keyword arguments.
        """
        pass

    def cleanup(self):
        """
        Optional method to clean up any residual state on termination.
        """
        pass

    def diagnostics(self):
        """
        Relevant diagnostics (optional) to be printed at regular intervals
        of the MCMC run. Returns `None` by default.

        :return: String containing the diagnostic summary. e.g. acceptance rate
        :rtype: string
        """
        return None

    def end_warmup(self):
        """
        Optional method to tell kernel that warm-up phase has been finished.
        """
        pass

    @abstractmethod
    def initial_trace(self):
        """
        Returns an initial trace from the prior to initiate the MCMC run.

        :return: Trace instance.
        """
        return NotImplementedError

    @abstractmethod
    def sample(self, trace):
        """
        Samples a trace from the approximate posterior distribution, when given an existing trace.

        :param trace: Current execution trace.
        :param int time_step: Current time step.
        :return: New trace sampled from the approximate posterior distribution.
        """
        raise NotImplementedError

    def __call__(self, trace):
        """
        Alias for TraceKernel.sample() method.
        """
        return self.sample(trace)
