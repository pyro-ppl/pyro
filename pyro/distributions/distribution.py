from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

from six import add_metaclass

from pyro.distributions.score_parts import ScoreParts


@add_metaclass(ABCMeta)
class Distribution(object):
    """
    Base class for parameterized probability distributions.

    Distributions in Pyro are stochastic function objects with :meth:`sample` and
    :meth:`log_prob` methods. Distribution are stochastic functions with fixed
    parameters::

      d = dist.Bernoulli(param)
      x = d()                                # Draws a random sample.
      p = d.log_prob(x)                      # Evaluates log probability of x.

    **Implementing New Distributions**:

    Derived classes must implement the methods: :meth:`sample`,
    :meth:`log_prob`.

    **Examples**:

    Take a look at the `examples <http://pyro.ai/examples>`_ to see how they interact
    with inference algorithms.
    """
    has_rsample = False
    has_enumerate_support = False

    def __call__(self, *args, **kwargs):
        """
        Samples a random value (just an alias for ``.sample(*args, **kwargs)``).

        For tensor distributions, the returned tensor should have the same ``.shape`` as the
        parameters.

        :return: A random value.
        :rtype: torch.Tensor
        """
        return self.sample(*args, **kwargs)

    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Samples a random value.

        For tensor distributions, the returned tensor should have the same ``.shape`` as the
        parameters, unless otherwise noted.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be ``self.shape()``.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x, *args, **kwargs):
        """
        Evaluates log probability densities for each of a batch of samples.

        :param torch.Tensor x: A single value or a batch of values
            batched along axis 0.
        :return: log probability densities as a one-dimensional
            :class:`~torch.Tensor` with same batch size as value and
            params. The shape of the result should be ``self.batch_size``.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def score_parts(self, x, *args, **kwargs):
        """
        Computes ingredients for stochastic gradient estimators of ELBO.

        The default implementation is correct both for non-reparameterized and
        for fully reparameterized distributions. Partially reparameterized
        distributions should override this method to compute correct
        `.score_function` and `.entropy_term` parts.

        :param torch.Tensor x: A single value or batch of values.
        :return: A `ScoreParts` object containing parts of the ELBO estimator.
        :rtype: ScoreParts
        """
        log_prob = self.log_prob(x, *args, **kwargs)
        if self.has_rsample:
            return ScoreParts(log_prob=log_prob, score_function=0, entropy_term=log_prob)
        else:
            # XXX should the user be able to control inclusion of the entropy term?
            # See Roeder, Wu, Duvenaud (2017) "Sticking the Landing" https://arxiv.org/abs/1703.09194
            return ScoreParts(log_prob=log_prob, score_function=log_prob, entropy_term=0)

    def enumerate_support(self):
        """
        Returns a representation of the parametrized distribution's support,
        along the first dimension. This is implemented only by discrete
        distributions.

        Note that this returns support values of all the batched RVs in
        lock-step, rather than the full cartesian product.

        :return: An iterator over the distribution's discrete support.
        :rtype: iterator
        """
        raise NotImplementedError("Support not implemented for {}".format(type(self)))
