import argparse
import torch
from torch.distributions import constraints
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_ape
import pyro.contrib.gp as gp

from gp_bayes_opt import GPBayesOptimizer

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
prior_stdevs = torch.tensor([10., 2.5])

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
    if t < n:
        X[t:, -1] = 1.
    return X


def analytic_posterior_entropy(prior_cov, x):
    # Use some kernel trick magic
    SigmaXX = prior_cov.mm(x.t().mm(x))
    posterior_cov = prior_cov - torch.inverse(
        SigmaXX + torch.eye(p)).mm(SigmaXX.mm(prior_cov))
    y = 0.5*torch.logdet(2*np.pi*np.e*posterior_cov)
    return y


def main(num_vi_steps, num_acquisitions, num_bo_steps):

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    def estimated_ape(ns):
        designs = [design_to_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
        X = torch.stack(designs)
        est_ape = vi_ape(
            model,
            X,
            observation_labels="y",
            vi_parameters={
                "guide": guide,
                "optim": optim.Adam({"lr": 0.0025}),
                "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
                "num_steps": num_vi_steps},
            is_parameters={"num_samples": 1}
        )
        return est_ape

    def true_ape(ns):
        true_ape = []
        prior_cov = torch.diag(prior_stdevs**2)
        designs = [design_to_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
        for i in range(len(ns)):
            x = designs[i]
            true_ape.append(analytic_posterior_entropy(prior_cov, x))
        return torch.tensor(true_ape)

    for f in [true_ape, estimated_ape]:
        X = torch.tensor([25., 75.])
        y = f(X)
        pyro.clear_param_store()
        gpmodel = gp.models.GPRegression(
            X, y, gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(5.)),
            noise=torch.tensor(0.1), jitter=1e-6)
        gpmodel.optimize(loss=TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss)
        gpbo = GPBayesOptimizer(constraints.interval(0, 100), gpmodel,
                                num_acquisitions=num_acquisitions)
        for i in range(num_bo_steps):
            result = gpbo.get_step(f, None)

        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test experiment design using VI")
    parser.add_argument("-n", "--num-vi-steps", nargs="?", default=5000, type=int)
    parser.add_argument('--num-acquisitions', nargs="?", default=10, type=int)
    parser.add_argument('--num-bo-steps', nargs="?", default=6, type=int)
    args = parser.parse_args()
    main(args.num_vi_steps, args.num_acquisitions, args.num_bo_steps)
