# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod


class MCMCKernel(object, metaclass=ABCMeta):

    def setup(self, warmup_steps, *args, **kwargs):
        r"""
        Optional method to set up any state required at the start of the
        simulation run.

        :param int warmup_steps: Number of warmup iterations.
        :param \*args: Algorithm specific positional arguments.
        :param \*\*kwargs: Algorithm specific keyword arguments.
        """
        pass

    def cleanup(self):
        """
        Optional method to clean up any residual state on termination.
        """
        pass

    def logging(self):
        """
        Relevant logging information to be printed at regular intervals
        of the MCMC run. Returns `None` by default.

        :return: String containing the diagnostic summary. e.g. acceptance rate
        :rtype: string
        """
        return None

    def diagnostics(self):
        """
        Returns a dict of useful diagnostics after finishing sampling process.
        """
        # NB: should be not None for multiprocessing works
        return {}

    def end_warmup(self):
        """
        Optional method to tell kernel that warm-up phase has been finished.
        """
        pass

    @property
    def initial_params(self):
        """
        Returns a dict of initial params (by default, from the prior) to initiate the MCMC run.

        :return: dict of parameter values keyed by their name.
        """
        raise NotImplementedError

    @initial_params.setter
    def initial_params(self, params):
        """
        Sets the parameters to initiate the MCMC run. Note that the parameters must
        have unconstrained support.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, params):
        """
        Samples parameters from the posterior distribution, when given existing parameters.

        :param dict params: Current parameter values.
        :param int time_step: Current time step.
        :return: New parameters from the posterior distribution.
        """
        raise NotImplementedError

    def __call__(self, params):
        """
        Alias for MCMCKernel.sample() method.
        """
        return self.sample(params)
