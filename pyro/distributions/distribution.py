from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.score_parts import ScoreParts


class Distribution(object):
    """
    Mixin to provide Pyro compatibility for PyTorch distributions.
    """
    @property
    def reparameterized(self):
        return self.has_rsample

    @property
    def enumerable(self):
        return self.has_enumerate_support

    def __call__(self, sample_shape=torch.Size()):
        """
        Samples a random value. This is reparameterized whenever possible.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be `self.shape()`.
        :rtype: torch.autograd.Variable
        """
        return self.rsample(sample_shape) if self.has_rsample else self.sample(sample_shape)

    @property
    def event_dim(self):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape)

    def shape(self, sample_shape=torch.Size()):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape :code:`d.shape(sample_shape) == sample_shape +
        d.batch_shape() + d.event_shape()`.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: Tensor shape of samples.
        :rtype: torch.Size
        """
        return sample_shape + self.batch_shape + self.event_shape

    def score_parts(self, x):
        """
        Computes ingredients for stochastic gradient estimators of ELBO.

        The default implementation is correct both for non-reparameterized and
        for fully reparameterized distributions. Partially reparameterized
        distributions should override this method to compute correct
        `.score_function` and `.entropy_term` parts.

        :param torch.autograd.Variable x: A single value or batch of values.
        :return: A `ScoreParts` object containing parts of the ELBO estimator.
        :rtype: ScoreParts
        """
        log_pdf = self.log_prob(x)
        if self.reparameterized:
            return ScoreParts(log_pdf=log_pdf, score_function=0, entropy_term=log_pdf)
        else:
            # XXX should the user be able to control inclusion of the entropy term?
            # See Roeder, Wu, Duvenaud (2017) "Sticking the Landing" https://arxiv.org/abs/1703.09194
            return ScoreParts(log_pdf=log_pdf, score_function=log_pdf, entropy_term=0)

    def analytic_mean(self):
        return self.mean

    def analytic_var(self):
        return self.variance
