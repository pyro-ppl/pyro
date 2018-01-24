from __future__ import absolute_import, division, print_function

from collections import namedtuple


class ScoreParts(namedtuple('ScoreParts', ['log_pdf', 'score_function', 'entropy_term'])):
    """
    This data structure stores terms used in stochastic gradient estimators that
    combine the pathwise estimator and the score function estimator.
    """
    def __mul__(self, scale):
        """
        Scale appropriate terms of a gradient estimator by a data multiplicity factor.
        Note that the `score_function` term should not be scaled.
        """
        log_pdf, score_function, entropy_term = self
        return ScoreParts(log_pdf * scale, score_function, entropy_term * scale)

    __rmul__ = __mul__
