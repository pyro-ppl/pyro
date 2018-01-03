from __future__ import absolute_import, division, print_function

from pyro.distributions.gamma import Gamma
from pyro.distributions.rejector import ImplicitRejector
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Gamma)
class RejectionGamma(ImplicitRejector):
    def __init__(self, alpha):
        assert (alpha >= 1).all()
        self.gamma = Gamma(alpha)
        super(RejectionGamma, self).__init__(self.gamma, self._acceptor)

    def _acceptor(self, x):
        return 'TODO: back out acceptance probability from Margsaglia & Tsang'

    def sample(self):
        detached_sample = self.gamma.sample().detach()
        zero_with_correct_grad = 'TODO back out partially reparam grad'
        return detached_sample + zero_with_correct_grad
