import argparse
import torch
from torch.distributions import constraints, transform_to
import torch.autograd as autograd
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import Trace_ELBO
from pyro.contrib.oed.eig import vi_ape
import pyro.contrib.gp as gp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

"""
Example builds on the Bayesian regression tutorial [1]. It demonstrates how
to estimate the average posterior entropy (APE) under a model and use it to
make an optimal decision about experiment design.

The context is a Gaussian linear model in which the design matrix `X` is a
one-hot-encoded matrix with 2 columns. This corresponds to the simplest form
of an A/B test. Assume no data has yet be collected. The aim is to find the optimal
allocation of participants to the two groups to maximise the expected gain in
information from actually performing the experiment.

For details of the implementation of average posterior entropy estimation, see
the docs for :func:`pyro.contrib.oed.eig.vi_ape`.

We recommend the technical report from Long Ouyang et al [3] as an introduction
to optimal experiment design within probabilistic programs.

[1] ["Bayesian Regression"](http://pyro.ai/examples/bayesian_regression.html)
[2] Long Ouyang, Michael Henry Tessler, Daniel Ly, Noah Goodman (2016),
    "Practical optimal experiment design with probabilistic programs",
    (https://arxiv.org/abs/1608.05046)
"""

# Set up regression model dimensions
N = 100  # number of participants
p_treatments = 2  # number of treatment groups
p = p_treatments  # number of features
prior_stdevs = torch.tensor([1, .5])

softplus = torch.nn.functional.softplus


def model(design):
    # Allow batching of designs
    loc_shape = list(design.shape)
    loc_shape[-2] = 1
    loc = torch.zeros(loc_shape)
    scale = prior_stdevs
    # Place a normal prior on the regression coefficient
    # w is 1 x p: hence use .independent(2)
    w_prior = dist.Normal(loc, scale).independent(2)
    w = pyro.sample('w', w_prior).transpose(-1, -2)

    # Run the regressor forward conditioned on inputs
    prediction_mean = torch.matmul(design, w).squeeze(-1)
    # y is an n-vector: hence use .independent(1)
    pyro.sample("y", dist.Normal(prediction_mean, 1).independent(1))


def guide(design):
    # Guide defines a Gaussian family for `w`
    # The true posterior for `w` is within this family
    # In this case, variational inference will be exact
    loc_shape = list(design.shape)
    loc_shape[-2] = 1
    # define our variational parameters
    w_loc = torch.zeros(loc_shape)
    # note that we initialize our scales to be pretty narrow
    w_sig = -3*torch.ones(loc_shape)
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_scale_weight", w_sig))
    # guide distributions for w
    w_dist = dist.Normal(mw_param, sw_param).independent(2)
    pyro.sample('w', w_dist)


def design_to_matrix(design):
    """Converts a one-dimensional tensor listing group sizes into a
    two-dimensional binary tensor of indicator variables.

    :return: A :math:`n \times p` binary matrix where :math:`p` is
        the length of `design` and :math:`n` is its sum. There are
        :math:`n_i` ones in the :math:`i`th column.
    :rtype: torch.tensor

    """
    n, p = int(torch.sum(design)), int(design.size()[0])
    X = torch.zeros(n, p)
    t = 0
    for col, i in enumerate(design):
        i = int(i)
        if i > 0:
            X[t:t+i, col] = 1.
        t += i
    return X


def analytic_posterior_entropy(prior_cov, x):
    posterior_cov = prior_cov - prior_cov.mm(x.t().mm(torch.inverse(
        x.mm(prior_cov.mm(x.t())) + torch.eye(N)).mm(x.mm(prior_cov))))
    return 0.5*torch.logdet(2*np.pi*np.e*posterior_cov)


def bayes_opt(f, num_steps=10):
    # initialize the model with some points
    X = torch.tensor([25., 75.])
    y = f(X)
    y.detach()
    gpmodel = gp.models.GPRegression(X, y, gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(5.)),
                                     noise=torch.tensor(0.1), jitter=1.0e-4)
    gpmodel.optimize()
    y -= torch.mean(y)
    print(X, y)

    def update_posterior(x_new):
        pyro.clear_param_store()
        y = f(x_new) # evaluate f at new point.
        X = torch.cat([gpmodel.X, x_new]) # incorporate new evaluation
        y = torch.cat([gpmodel.y, y])
        y -= torch.mean(y)
        gpmodel.set_data(X, y)
        gpmodel.optimize()  # optimize the GP hyperparameters using default settings
        return X, y

    def lower_confidence_bound(x, kappa=2):
        mu, variance = gpmodel(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - kappa * sigma

    def find_a_candidate(x_init, lower_bound=0, upper_bound=1):
        # transform x to an unconstrained domain
        constraint = constraints.interval(lower_bound, upper_bound)
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        unconstrained_x = torch.tensor(unconstrained_x_init, requires_grad=True)
        minimizer = torch.optim.LBFGS([unconstrained_x])

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = lower_confidence_bound(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(constraint)(unconstrained_x)
        return x.detach()

    def next_x(lower_bound=0, upper_bound=1, num_candidates=5):
        candidates = []
        values = []

        x_init = gpmodel.X[-1:]
        for i in range(num_candidates):
            x = find_a_candidate(x_init, lower_bound, upper_bound)
            y = lower_confidence_bound(x)
            candidates.append(x)
            values.append(y)
            x_init = x.new_empty(1).uniform_(lower_bound, upper_bound)

        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        return candidates[argmin]

    def plot(gs, xmin, xlabel=None, with_title=True):
        xlabel = "xmin" if xlabel is None else "x{}".format(xlabel)
        Xnew = torch.linspace(-1., 101.)
        ax1 = plt.subplot(gs[0])
        ax1.plot(list(gpmodel.X), list(gpmodel.y), "kx")  # plot all observed data
        with torch.no_grad():
            loc, var = gpmodel(Xnew, full_cov=False, noiseless=False)
            sd = var.sqrt()
            ax1.plot(Xnew.numpy(), loc.numpy(), "r", lw=2)  # plot predictive mean
            ax1.fill_between(Xnew.numpy(), loc.numpy() - 2*sd.numpy(), loc.numpy() + 2*sd.numpy(),
                             color="C0", alpha=0.3)  # plot uncertainty intervals
        ax1.set_xlim(-1., 101.)
        ax1.set_title("Find {}".format(xlabel))
        if with_title:
            ax1.set_ylabel("Gaussian Process Regression")

        ax2 = plt.subplot(gs[1])
        with torch.no_grad():
            # plot the acquisition function
            ax2.plot(Xnew.numpy(), lower_confidence_bound(Xnew).numpy())
            # plot the new candidate point
            ax2.plot(xmin.numpy(), lower_confidence_bound(xmin).numpy(), "^", markersize=10,
                     label="{} = {:.5f}".format(xlabel, xmin.item()))
        ax2.set_xlim(-1., 101.)
        if with_title:
            ax2.set_ylabel("Acquisition Function")
        ax2.legend(loc=1)

    plt.figure(figsize=(12, 20))
    outer_gs = gridspec.GridSpec(5, 2)
    gpmodel.optimize()
    for i in range(num_steps):
        xmin = next_x(upper_bound=100)
        print(xmin)
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i])
        plot(gs, xmin, xlabel=i+1, with_title=(i % 2 == 0))
        update_posterior(xmin)
    plt.show()





def main(num_steps):

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    def f(ns):
        designs = [design_to_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
        X = torch.stack(designs)
        est_ape = vi_ape(
            model,
            X,
            observation_labels="y",
            vi_parameters={
                "guide": guide, 
                "optim": optim.Adam({"lr": 0.0025}),
                "loss": Trace_ELBO(),
                "num_steps": num_steps},
            is_parameters={"num_samples": 2}
        )
        return est_ape

    bayes_opt(f)

    # # Estimated loss (linear transform of EIG)
    # est_ape = vi_ape(
    #     model,
    #     X,
    #     observation_labels="y",
    #     vi_parameters={
    #         "guide": guide, "optim": optim.Adam({"lr": 0.0025}),
    #         "num_steps": num_steps},
    #     is_parameters={"num_samples": 2}
    # )

    # # Analytic loss
    # true_ape = []
    # prior_cov = torch.diag(prior_stdevs**2)
    # for i in range(len(ns)):
    #     x = X[i, :, :]
    #     true_ape.append(analytic_posterior_entropy(prior_cov, x))
    # true_ape = torch.tensor(true_ape)

    # print("Estimated APE values")
    # print(est_ape)
    # print("True APE values")
    # print(true_ape)

    # # Plot to compare
    # import matplotlib.pyplot as plt
    # ns = np.array(ns)
    # est_ape = np.array(est_ape.detach())
    # true_ape = np.array(true_ape)
    # plt.scatter(ns, est_ape)
    # plt.scatter(ns, true_ape, color='r')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test experiment design using VI")
    parser.add_argument("-n", "--num-steps", nargs="?", default=3000, type=int)
    args = parser.parse_args()
    main(args.num_steps)
