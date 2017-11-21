import pyro
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
        super(Laplace, self).__init__(model, optim, *args, **kwargs)

    def get_hessians(self, params, *args, **kwargs):
        """
        :param xs: Parameters' name
        :returns: A dict of the form `param: Hessian`
        """
        xs = [pyro.param(p) for p in params]
        kwargs["callback"] = (hessians, xs)
        hs = self.loss_and_grads(self.model, self.guide, *args, **kwargs)
        return dict(zip(params, hs))
