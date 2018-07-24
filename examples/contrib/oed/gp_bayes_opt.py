import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import transform_to

import pyro


class GPBayesOptimizer(pyro.optim.multi.MultiOptimizer):
    """Performs Bayesian Optimization using a Gaussian Process as an
    emulator for the unknown function.
    """

    def __init__(self, constraints, gpmodel, num_acquisitions, acquisition_func=None):
        """
        :param torch.constraint constraints: constraints defining the domain of `f`
        :param gp.models.GPRegression gpmodel: a (possibly initialized) GP
            regression model. The kernel, etc is specified via `gpmodel`.
        :param int num_acquisitions: number of points to acquire at each step
        :param function acquisition_func: a function to generate acquisitions.
            It should return a torch.Tensor of new points to query.
        """
        if acquisition_func is None:
            acquisition_func = self.acquire_thompson

        self.constraints = constraints
        self.gpmodel = gpmodel
        self.num_acquisitions = num_acquisitions
        self.acquisition_func = acquisition_func

    def update_posterior(self, X, y):
        X = torch.cat([self.gpmodel.X, X])
        y = torch.cat([self.gpmodel.y, y])
        self.gpmodel.set_data(X, y)
        self.gpmodel.optimize()

    def find_a_candidate(self, differentiable, x_init):
        """Given a starting point, `x_init`, takes one LBFGS step
        to optimize the differentiable function.

        :param function differentiable: a function amenable to torch
            autograd
        :param torch.Tensor x_init: the initial point

        """
        # transform x to an unconstrained domain
        unconstrained_x_init = transform_to(self.constraints).inv(x_init)
        unconstrained_x = unconstrained_x_init.new_tensor(
            unconstrained_x_init, requires_grad=True)
        # TODO: Use LBFGS with line search by pytorch #8824 merged
        minimizer = optim.LBFGS([unconstrained_x], max_eval=20)

        def closure():
            minimizer.zero_grad()
            if (torch.log(torch.abs(unconstrained_x)) > 25.).any():
                return torch.tensor(float('inf'))
            x = transform_to(self.constraints)(unconstrained_x)
            y = differentiable(x)
            autograd.backward(unconstrained_x,
                              autograd.grad(y, unconstrained_x, retain_graph=True))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(self.constraints)(unconstrained_x)
        opt_y = differentiable(x)
        return x.detach(), opt_y.detach()

    def opt_differentiable(self, differentiable, num_candidates=5):
        """Optimizes a differentiable function by choosing `num_candidates`
        initial points at random and calling :func:`find_a_candidate` on
        each. The best candidate is returned with its function value.

        :param function differentiable: a function amenable to torch autograd
        :param int num_candidates: the number of random starting points to
            use
        :return: the minimiser and its function value
        :rtype: tuple
        """

        candidates = []
        values = []
        for j in range(num_candidates):
            x_init = self.gpmodel.X.new_empty(1).uniform_(
                self.constraints.lower_bound, self.constraints.upper_bound)
            x, y = self.find_a_candidate(differentiable, x_init)
            candidates.append(x)
            values.append(y)

        mvalue, argmin = torch.min(torch.cat(values), dim=0)
        return candidates[argmin.item()], mvalue

    def acquire_thompson(self, num_acquisitions=1, **opt_params):
        """Selects `num_acquisitions` query points at which to query the
        original function by Thompson sampling.

        :param int num_acquisitions: the number of points to generate
        :param dict opt_params: additional parameters for optimization
            routines
        :return: a tensor of points to evaluate `self.f` at
        :rtype: torch.Tensor
        """

        # Initialize the return tensor
        X = self.gpmodel.X.new_empty(num_acquisitions, *self.gpmodel.X.shape[1:])

        for i in range(num_acquisitions):
            sampler = self.gpmodel.iter_sample(noiseless=False)
            x, _ = self.opt_differentiable(sampler, **opt_params)
            X[i, ...] = x

        return X

    def get_step(self, loss, params):
        X = self.acquisition_func(num_acquisitions=self.num_acquisitions)
        y = loss(X)
        self.update_posterior(X, y)
        return self.opt_differentiable(lambda x: self.gpmodel(x)[0])
