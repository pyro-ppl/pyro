import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import transform_to


class GPBayesOptimizer:
    """Performs Bayesian Optimization using a Gaussian Process as an
    emulator for the unknown function.
    """

    def __init__(self, f, constraints, gpmodel):
        """
        :param function f: the objective function which should accept `torch.Tensor`
            inputs
        :param torch.constraint constraints: constraints defining the domain of `f`
        :param gp.models.GPRegression gpmodel: a (possibly initialized) GP
            regression model. The kernel, etc is specified via `gpmodel`.
        """
        self.f = f
        self.constraints = constraints
        self.gpmodel = gpmodel

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
        unconstrained_x = torch.tensor(unconstrained_x_init, requires_grad=True)
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

        x_init = self.gpmodel.X[-1:].new_empty(1).uniform_(self.constraints.lower_bound,
                                                           self.constraints.upper_bound)
        candidates = []
        values = []
        for j in range(num_candidates):
            x, y = self.find_a_candidate(differentiable, x_init)
            candidates.append(x)
            values.append(y)
            x_init = x.new_empty(1).uniform_(self.constraints.lower_bound,
                                             self.constraints.upper_bound)

        mvalue, argmin = torch.min(torch.cat(values), dim=0)
        return candidates[argmin.item()], mvalue

    def acquire(self, method="Thompson", num_acquisitions=1, **opt_params):
        """Selects `num_acquisitions` query points at which to query the
        original function.

        :param str method: the method to use for acquisition. Choose from:
            Thompson
        :param int num_acquisitions: the number of points to generate
        :param dict opt_params: additional parameters for optimization
            routines
        :return: a tensor of points to evaluate `self.f` at
        :rtype: torch.Tensor
        """

        if method == "Thompson":

            # Initialize the return tensor, add the UCB1 point
            X = torch.zeros(num_acquisitions, *self.gpmodel.X.shape[1:])

            for i in range(num_acquisitions):
                sampler = self.gpmodel.iter_sample(noiseless=False)
                x, _ = self.opt_differentiable(sampler, **opt_params)
                X[i, ...] = x

            return X

        else:
            raise NotImplementedError("Only method Thompson implemented for acquisition")

    # def run(self, num_steps, num_acquisitions):
    #     """
    #     Optimizes `self.f` in `num_steps` steps, acquiring `num_acquisitions`
    #     new function evaluations at each step.

    #     :param int num_steps:
    #     :param int num_steps"
    #     :return: the minimiser and the minimum value
    #     :rtype: tuple
    #     """

    #     for i in range(num_steps):
    #         X = self.acquire(num_acquisitions=num_acquisitions)
    #         y = self.f(X)
    #         self.update_posterior(X, y)

    #     return self.opt_differentiable(lambda x: self.gpmodel(x)[0])

    def run(self, num_steps, num_acquisitions):
        plt.figure(figsize=(12, 30))
        # outer_gs = gridspec.GridSpec(num_steps, 1)

        for i in range(num_steps):
            X = self.acquire(num_acquisitions=num_acquisitions)
            y = self.f(X)
            # gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[i])
            plt.figure()
            self.plot(plt.gca(), xlabel=i+1, with_title=True)
            self.update_posterior(X, y)

            plt.show()

        return self.opt_differentiable(lambda x: self.gpmodel(x)[0])

    def plot(self, ax1, xlabel=None, with_title=True):
        xlabel = str(xlabel)
        Xnew = torch.linspace(-1., 101.)
        # ax1 = plt.subplot(gs[0])
        ax1.plot(self.gpmodel.X.detach().numpy(), self.gpmodel.y.detach().numpy(), "kx")  # plot all observed data
        with torch.no_grad():
            loc, var = self.gpmodel(Xnew, full_cov=False, noiseless=False)
            sd = var.sqrt()
            ax1.plot(Xnew.numpy(), loc.numpy(), "r", lw=2)  # plot predictive mean
            ax1.fill_between(Xnew.numpy(), loc.numpy() - 2*sd.numpy(), loc.numpy() + 2*sd.numpy(),
                             color="C0", alpha=0.3)  # plot uncertainty intervals
        ax1.set_xlim(-1, 101)
        ax1.set_title("Find design {}".format(xlabel))
        if with_title:
            ax1.set_ylabel("EIG estimate")
