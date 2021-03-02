# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
from abc import ABCMeta, abstractmethod

from pyro.distributions.score_parts import ScoreParts

COERCIONS = []


class DistributionMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        for coerce_ in COERCIONS:
            result = coerce_(cls, args, kwargs)
            if result is not None:
                return result
        return super().__call__(*args, **kwargs)

    @property
    def __wrapped__(cls):
        return functools.partial(cls.__init__, None)


class Distribution(metaclass=DistributionMeta):
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

        Setting ``.has_rsample`` on a distribution instance will determine
        whether inference engines like :class:`~pyro.infer.svi.SVI` use
        reparameterized samplers or the score function estimator.

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

    def enumerate_support(self, expand=True):
        """
        Returns a representation of the parametrized distribution's support,
        along the first dimension. This is implemented only by discrete
        distributions.

        Note that this returns support values of all the batched RVs in
        lock-step, rather than the full cartesian product.

        :param bool expand: whether to expand the result to a tensor of shape
            ``(n,) + batch_shape + event_shape``. If false, the return value
            has unexpanded shape ``(n,) + (1,)*len(batch_shape) + event_shape``
            which can be broadcasted to the full shape.
        :return: An iterator over the distribution's discrete support.
        :rtype: iterator
        """
        raise NotImplementedError("Support not implemented for {}".format(type(self).__name__))

    def conjugate_update(self, other):
        """
        EXPERIMENTAL Creates an updated distribution fusing information from
        another compatible distribution. This is supported by only a few
        conjugate distributions.

        This should satisfy the equation::

            fg, log_normalizer = f.conjugate_update(g)
            assert f.log_prob(x) + g.log_prob(x) == fg.log_prob(x) + log_normalizer

        Note this is equivalent to :obj:`funsor.ops.add` on
        :class:`~funsor.terms.Funsor` distributions, but we return a lazy sum
        ``(updated, log_normalizer)`` because PyTorch distributions must be
        normalized.  Thus :meth:`conjugate_update` should commute with
        :func:`~funsor.pyro.convert.dist_to_funsor` and
        :func:`~funsor.pyro.convert.tensor_to_funsor` ::

            dist_to_funsor(f) + dist_to_funsor(g)
              == dist_to_funsor(fg) + tensor_to_funsor(log_normalizer)

        :param other: A distribution representing ``p(data|latent)`` but
            normalized over ``latent`` rather than ``data``. Here ``latent``
            is a candidate sample from ``self`` and ``data`` is a ground
            observation of unrelated type.
        :return: a pair ``(updated,log_normalizer)`` where ``updated`` is an
            updated distribution of type ``type(self)``, and ``log_normalizer``
            is a :class:`~torch.Tensor` representing the normalization factor.
        """
        raise NotImplementedError("{} does not support .conjugate_update()"
                                  .format(type(self).__name__))

    def has_rsample_(self, value):
        """
        Force reparameterized or detached sampling on a single distribution
        instance. This sets the ``.has_rsample`` attribute in-place.

        This is useful to instruct inference algorithms to avoid
        reparameterized gradients for variables that discontinuously determine
        downstream control flow.

        :param bool value: Whether samples will be pathwise differentiable.
        :return: self
        :rtype: Distribution
        """
        if not (value is True or value is False):
            raise ValueError("Expected value in [False,True], actual {}".format(value))
        self.has_rsample = value
        return self

    @property
    def rv(self):
        """
        EXPERIMENTAL Switch to the Random Variable DSL for applying transformations
        to random variables. Supports either chaining operations or arithmetic
        operator overloading.

        Example usage::

            # This should be equivalent to an Exponential distribution.
            Uniform(0, 1).rv.log().neg().dist

            # These two distributions Y1, Y2 should be the same
            X = Uniform(0, 1).rv
            Y1 = X.mul(4).pow(0.5).sub(1).abs().neg().dist
            Y2 = (-abs((4*X)**(0.5) - 1)).dist


        :return: A :class: `~pyro.contrib.randomvariable.random_variable.RandomVariable`
            object wrapping this distribution.
        :rtype: ~pyro.contrib.randomvariable.random_variable.RandomVariable
        """
        from pyro.contrib.randomvariable import RandomVariable
        return RandomVariable(self)
