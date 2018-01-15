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
        gammas = self._gamma.sample()
        return gammas / gammas.sum(-1, True)


@copy_docs_from(Dirichlet)
class ShapeAugmentedDirichlet(Dirichlet):
    def __init__(self, alpha, boost=1, allow_bias=False, *args, **kwargs):
        beta = alpha.new([1]).expand_as(alpha)
        self._gamma = ShapeAugmentedGamma(alpha, beta, boost=boost, *args, **kwargs)
        self.allow_bias = allow_bias
        super(ShapeAugmentedDirichlet, self).__init__(alpha, *args, **kwargs)

    def sample(self):
        gammas = self._gamma.sample()
        return gammas / gammas.sum(-1, True)

    def score_parts(self, x):
        log_pdf = self.batch_log_pdf(x)
        if self.allow_bias:
            score_function = 0
        else:
            score_function = self._gamma.score_parts()[1].sum(-1, True)
            assert score_function.shape == log_pdf.shape, (score_function.shape, log_pdf.shape)
        entropy_part = -self.entropy()  # as in RSVI paper
        assert entropy_part.shape == log_pdf.shape, (entropy_part.shape, log_pdf.shape)
        return ScoreParts(log_pdf, score_function, entropy_part)
