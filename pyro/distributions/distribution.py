from __future__ import absolute_import, division, print_function

from pyro.distributions.score_parts import ScoreParts


class Distribution(object):
    """
    Abstract base class for Pyro probability distributions.
    """
    reparameterized = False
    enumerable = False

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def log_prob(self, x, *args, **kwargs):
        raise NotImplementedError

    def enumerate_support(self, *args, **kwargs):
        raise NotImplementedError

    def score_parts(self, x, *args, **kwargs):
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
        log_pdf = self.log_prob(x, *args, **kwargs)
        if self.reparameterized:
            return ScoreParts(log_pdf=log_pdf, score_function=0, entropy_term=log_pdf)
        else:
            # XXX should the user be able to control inclusion of the entropy term?
            # See Roeder, Wu, Duvenaud (2017) "Sticking the Landing" https://arxiv.org/abs/1703.09194
            return ScoreParts(log_pdf=log_pdf, score_function=log_pdf, entropy_term=0)
