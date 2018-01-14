from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedGamma
from pyro.distributions.torch.dirichlet import Dirichlet
from pyro.distributions.torch.gamma import Gamma
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Dirichlet)
class GammaDirichlet(Dirichlet):
    """
    Implementation of Dirichlet distribution using reparameterized Gammas.

    This results in stochatsic reparameterized gradients that are only correct
    in expecation.
    """
    def __init__(self, alpha, *args, **kwargs):
        self._gamma = Gamma(alpha, alpha.new([1]).expand_as(alpha), *args, **kwargs)
        super(GammaDirichlet, self).__init__(alpha, *args, **kwargs)

    def sample(self):
        probs = self._gamma.sample()
        probs /= probs.sum(-1, True)
        return probs


@copy_docs_from(Dirichlet)
class ShapeAugmentedDirichlet(Dirichlet):
    def __init__(self, alpha, boost=1, *args, **kwargs):
        beta = alpha.new([1]).expand_as(alpha)
        self._gamma = ShapeAugmentedGamma(alpha, beta, boost=boost, *args, **kwargs)
        super(ShapeAugmentedDirichlet, self).__init__(alpha, *args, **kwargs)

    def sample(self):
        probs = self._gamma.sample()
        probs /= probs.sum(-1, True)
        return probs

    def score_parts(self, x):
        entropy_part = -self.entropy()  # as in RSVI paper
        log_pdf = self.batch_log_pdf(x)
        assert entropy_part.shape == log_pdf.shape, (entropy_part.shape, log_pdf.shape)
        return ScoreParts(log_pdf, 0, entropy_part)
