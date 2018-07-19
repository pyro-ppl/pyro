import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.nn import Parameter
from torch.distributions import constraints, transform_to
import copy

import pyro
import pyro.contrib.gp as gp


class GPBayesOptimizer:

    def __init__(self, f, constraints, gpmodel):
        self.f = f
        self.constraints = constraints
        self.gpmodel = gpmodel

    def update_posterior(self, gpmodel, X, y):
        X = torch.cat([gpmodel.X, X])
        y = torch.cat([gpmodel.y, y])
        gpmodel.set_data(X, y)
        gpmodel.optimize()

    def lower_confidence_bound(self, gpmodel, x, kappa=2.):
        mu, variance = gpmodel(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - kappa * sigma

    def find_a_candidate(self, gpmodel, x_init, **kwargs):
        # transform x to an unconstrained domain
        unconstrained_x_init = transform_to(self.constraints).inv(x_init)
        unconstrained_x = torch.tensor(unconstrained_x_init, requires_grad=True)
        minimizer = optim.LBFGS([unconstrained_x])

        def closure():
            minimizer.zero_grad()
            x = transform_to(self.constraints)(unconstrained_x)
            y = self.lower_confidence_bound(gpmodel, x, **kwargs)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(self.constraints)(unconstrained_x)
        return x.detach()

    def opt_lcb(self, num_candidates=5, kappa=2.):

        x_init = self.gpmodel.X[-1:].new_empty(1).uniform_(self.constraints.lower_bound, 
                    self.constraints.upper_bound)
        candidates = []
        values = []
        for j in range(num_candidates):
            x = self.find_a_candidate(self.gpmodel, x_init)
            y = self.lower_confidence_bound(self.gpmodel, x, kappa=kappa)
            candidates.append(x)
            values.append(y)
            x_init = x.new_empty(1).uniform_(self.constraints.lower_bound, 
                    self.constraints.upper_bound)

        mvalue, argmin = torch.min(torch.cat(values), dim=0)
        return candidates[argmin.item()], mvalue

    def acquire(self, method="MaxLB", num_candidates=1, num_acquisitions=1):

        if method == "MaxLB":

            # Minimise the upper bound
            ubm, maxvalue = self.opt_lcb(kappa=-2.)
            minvalue = self.lower_confidence_bound(self.gpmodel, ubm, kappa=2.)
            print(maxvalue)

            # Initialize the return tensor, add the UCB1 point
            X = torch.zeros(num_acquisitions, *self.gpmodel.X.shape[1:])
            X[0, ...], _ = self.opt_lcb(kappa=2.)
            
            nr = num_acquisitions - 1
            i = 1
            while nr > 0:
                x = X[0, ...].new_empty(1).uniform_(self.constraints.lower_bound, 
                    self.constraints.upper_bound)
                lb = self.lower_confidence_bound(self.gpmodel, x, kappa=2.)
                if lb <= maxvalue:
                    acceptance_probability = (maxvalue - lb)/(maxvalue - minvalue)
                    p = torch.rand((1,))[0]
                    if p <= acceptance_probability:
                        print('accept')
                        X[i, ...] = x
                        nr -= 1
                        i += 1
                    else:
                        print('rreject')
                else:
                    print('reject')
            print(X)
            return X

        else:
            raise NotImplementedError("Only method MaxLB implemented for acquisition")

    def run(self, num_steps, num_acquisitions):
        plt.figure(figsize=(12, 30))
        outer_gs = gridspec.GridSpec(num_steps, 1)

        for i in range(num_steps):
            X = self.acquire(num_acquisitions=num_acquisitions)
            y = self.f(X)
            gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[i])
            self.plot(gs, xlabel=i+1, with_title=(i % 2 == 0))
            self.update_posterior(self.gpmodel, X, y)

        plt.show()
        
        return self.opt_lcb(kappa=0.)


    def plot(self, gs, xlabel=None, with_title=True):
        xlabel = "xmin" if xlabel is None else "x{}".format(xlabel)
        Xnew = torch.linspace(-1., 101.)
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.gpmodel.X.detach().numpy(), self.gpmodel.y.detach().numpy(), "kx")  # plot all observed data
        with torch.no_grad():
            loc, var = self.gpmodel(Xnew, full_cov=False, noiseless=False)
            sd = var.sqrt()
            ax1.plot(Xnew.numpy(), loc.numpy(), "r", lw=2)  # plot predictive mean
            ax1.fill_between(Xnew.numpy(), loc.numpy() - 2*sd.numpy(), loc.numpy() + 2*sd.numpy(),
                             color="C0", alpha=0.3)  # plot uncertainty intervals
        ax1.set_xlim(-1, 101)
        ax1.set_title("Find {}".format(xlabel))
        if with_title:
            ax1.set_ylabel("Gaussian Process Regression")





