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
        
        
    def get_hessians(params, *args, **kwargs):
        """
        :param xs: Parameters' name
        :returns: Hessians of posteriors of each `x` in `xs`
        """
        xs = [pyro.param(p) for p in params]
        # callback is a tuple with first element is a function, and remaining elements is some of its arguments
        loss, hs = self.loss_and_grads(self.model, self.guide, callback=(hessians, xs), *args, **kwargs)
        hs = accumulate(hs)
        return dict(zip(params, hs))
