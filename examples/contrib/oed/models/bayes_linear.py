import warnings
import torch
from torch.nn.functional import softplus
import numpy as np

import pyro
import pyro.distributions as dist


def bayesian_linear_model(design, w_mean, w_sqrtlambda, obs_sd=None,
                          alpha_0=None, beta_0=None):
    """A Bayesian linear model.

    If `obs_sd` is passed, the regression coefficient `w` is samples from
    a Gaussian with mean `w_mean` and sds `obs_sd / w_sqrtlambda`.
    These tensors may be scalar or `p`-dimensional vectors.
    `X` is then sampled from a Gaussian with mean `Xw` and sd `obs_sd`.

    If `obs_sd=None`, the observation variance is sampled from an inverse
    Gamma distribution with parameters `alpha_0` and `beta_0`.
    Then `w` is sampled from a Gaussian with mean `w_mean` and sds given by
    `obs_sd / w_sqrtlambda` (cf. `lambda` in the NIG family).
    Finally, `X` is Gaussian with mean `Xw` and sd `sigma`.
    """
    if obs_sd is None:
        # First, sample tau (observation precision)
        tau_shape = design.shape[:-2]
        # Global variable
        tau_prior = dist.Gamma(alpha_0.expand(tau_shape),
                               beta_0.expand(tau_shape))
        tau = pyro.sample("tau", tau_prior)
        obs_sd = 1./torch.sqrt(tau).unsqueeze(-1)

    elif alpha_0 is not None or beta_0 is not None:
        warnings.warn("Values of `alpha_0` and `beta_0` unused becased"
                      "`obs_sd` was specified already.")

    # Allow batching of designs
    # design is batch x n x p
    loc_shape = list(design.shape)
    loc_shape[-2] = 1
    # loc is batch x 1 x p
    loc = w_mean.expand(loc_shape)
    # Place a normal prior on the regression coefficient
    w_prior = dist.Normal(loc, obs_sd.unsqueeze(-1) / w_sqrtlambda).independent(2)
    w = pyro.sample('w', w_prior).transpose(-1, -2)

    # Run the regressor forward conditioned on inputs
    prediction_mean = torch.matmul(design, w).squeeze(-1)
    # y is an n-vector: hence use .independent(1)
    pyro.sample("y", dist.Normal(prediction_mean, obs_sd).independent(1))


def normal_inv_gamma_guide(design, obs_sd):
    """Normal inverse Gamma family guide.

    If `obs_sd` is known, this is a two-parameter family with separate parameters
    for each batch. `w` is sampled from a Gaussian with mean `mw_param` and
    sd `obs_sd / lambda_param` and the two parameters `mw_param` and `lambda_param`
    are learned.

    If `obs_sd=None`, this is a four-parameter family. The observation precision
    `tau` is sampled from a Gamma distribution with parameters `alpha`, `beta`
    (separate for each batch). We let `obs_sd = 1./torch.sqrt(tau)` and then
    proceed as above.
    """
    if obs_sd is None:
        # First, sample tau (observation precision)
        tau_shape = design.shape[:-2]
        alpha = softplus(pyro.param("invsoftplus_alpha", 3.*torch.ones(tau_shape)))
        beta = softplus(pyro.param("invsoftplus_beta", 3.*torch.ones(tau_shape)))
        # Global variable
        tau_prior = dist.Gamma(alpha, beta)
        tau = pyro.sample("tau", tau_prior)
        obs_sd = 1./torch.sqrt(tau).unsqueeze(-1)

    loc_shape = list(design.shape)
    loc_shape[-2] = 1

    # Set up mu and lambda
    mw_param = pyro.param("guide_mean", torch.zeros(loc_shape))
    sqrtlambda_param = softplus(pyro.param("guide_sqrtlambda",
                                           3.*torch.ones(loc_shape)))
    # guide distributions for w
    w_dist = dist.Normal(mw_param, obs_sd.unsqueeze(-1) / sqrtlambda_param).independent(2)
    pyro.sample('w', w_dist)


def group_assignment_matrix(design):
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


def analytic_posterior_entropy(prior_cov, x, obs_sd):
    """
    Given a prior covariance matrix and a design matrix `x`,
    returns the entropy of the posterior under a Bayesian
    linear regression model with design `x` and observation
    noise `obs_sd`.
    """
    # Use some kernel trick magic
    p = prior_cov.shape[-1]
    SigmaXX = prior_cov.mm(x.t().mm(x))
    posterior_cov = prior_cov - torch.inverse(
        SigmaXX + (obs_sd**2)*torch.eye(p)).mm(SigmaXX.mm(prior_cov))
    return 0.5*torch.logdet(2*np.pi*np.e*posterior_cov)
