from __future__ import absolute_import, division, print_function

from abc import abstractmethod, ABCMeta

from six import add_metaclass


@add_metaclass(ABCMeta)
class TraceKernel(object):

    def setup(self, *args, **kwargs):
        """
        Optional method to set up any state required before the run.

        :param \*args: Algorithm specific positional arguments.
        :param \*\*kwargs: Algorithm specific keyword arguments.
        """
        pass

    def cleanup(self):
        """
        Optional method to clean up any residual state on termination.
        """
        pass

    def diagnostics(self, time_step):
        """
        Relevant diagnostics (optional) to be printed at regular intervals
        of the MCMC run. Returns `None` by default.

        :param time_step: Current Monte Carlo time-step.
        :return: String containing the diagnostic summary. e.g. acceptance rate
        :rtype: string
        """
        return None

    @abstractmethod
    def sample(self, trace, time_step, *args, **kwargs):
        """
        Samples a trace from the approximate posterior distribution, when given an existing trace.

        :param trace: Current execution trace.
        :param int time_step: Current time step.
        :param \*args: Algorithm specific positional arguments.
        :param \*\*kwargs: Algorithm specific keyword arguments.
        :return: New trace sampled from the approximate posterior distribution.
        """
        raise NotImplementedError
