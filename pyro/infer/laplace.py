import pyro
from pyro import poutine
from pyro.infer.map import MAP
from pyro.util import hessians


class Laplace(MAP):
    """
    :param model: the model (callable containing Pyro primitives)
    :param optim: a wrapper for a PyTorch optimizer
    :type optim: pyro.optim.PyroOptim

    Laplace approximation.
    """
    def __init__(self, model, optim, *args, **kwargs):
        model = poutine.lower(model)

        def guide(*args, **kwargs):
            pass

        super(MAP, self).__init__(model, guide, optim, loss="ELBO", *args, **kwargs)

    def get_hessians(self, params, data):
        """
        :param xs: Parameters' name
        :returns: A dict of the form `param: Hessian`
        """
        xs = [pyro.param(p) for p in params]
        hs = self.loss_and_grads(self.model, self.guide, data, callback=(hessians, xs))
        return dict(zip(params, hs))
