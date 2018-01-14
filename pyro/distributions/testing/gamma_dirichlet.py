from pyro.distributions.torch.dirichlet import Dirichlet
from pyro.distributions.torch.gamma import Gamma


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
