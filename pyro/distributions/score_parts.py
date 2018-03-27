from __future__ import absolute_import, division, print_function

from collections import namedtuple

from pyro.distributions.util import is_identically_one, scale_tensor, torch_sign


class ScoreParts(namedtuple('ScoreParts', ['log_prob_sum', 'score_function', 'entropy_term'])):
    """
    This data structure stores terms used in stochastic gradient estimators that
    combine the pathwise estimator and the score function estimator.
    """
    def __mul__(self, scale):
        """
        Scale appropriate terms of a gradient estimator by a data multiplicity factor.
        Note that the `score_function` term should not be scaled.
        """
        if is_identically_one(scale):
            return self
        log_prob_sum = scale_tensor(self.log_prob_sum, scale)
        score_function = scale_tensor(self.score_function, torch_sign(scale))
        entropy_term = scale_tensor(self.entropy_term, scale)
        return ScoreParts(log_prob_sum, score_function, entropy_term)

    __rmul__ = __mul__
