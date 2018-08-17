import warnings
import torch
from torch.nn.functional import softplus
import numpy as np

import pyro
import pyro.distributions as dist


def bayesian_linear_model(design, w_means=None, w_sqrtlambdas=None, obs_sd=None,
                          alpha_0=None, beta_0=None, response="normal",
                          response_label="y"):
    """
    A pyro model for Bayesian linear regression.

    If :param:`response` is `"normal"` this corresponds to a linear regression
    model

        :math:`Y = Xw + \\epsilon`

    with `\\epsilon`` i.i.d. zero-mean Gaussian. The observation standard deviation
    (:param:`obs_sd`) may be known or unknown. If unknown, it is assumed to follow an
    inverse Gamma distribution with parameters :param:`alpha_0` and :param:`beta_0`.

    If the response type is `"bernoulli"` we instead have :math:`Y \\sim Bernoulli(p)`
    with

        :math:`logit(p) = Xw`

    Given parameter groups in :param:`w_means` and :param:`w_sqrtlambda`, the regression
    coefficient is taken to be Gaussian with mean `w_mean` and standard deviation
    given by

        :math:`\\sigma / \\sqrt{\\lambda}`

    corresponding to the normal inverse Gamma family.

    :param torch.Tensor design: a tensor with last two dimensions `n` and `p`
        corresponding to observations and features respectively.
    :param dict w_means: map from variable names to tensors of parameter means.
    :param dict w_sqrtlambdas: map from variable names to tensors of square root
        :math:`\\lambda` values.
    :param torch.Tensor obs_sd: the observation standard deviation (if assumed known).
        This is still relevant in the case of Bernoulli observations when coefficeints
        are sampled using `w_sqrtlambdas`.
    :param torch.Tensor alpha_0: Gamma :math:`\\alpha` parameter for unknown observation
        covariance.
    :param torch.Tensor beta_0: Gamma :math:`\\beta` parameter for unknown observation
        covariance.
    :param str response: Emission distribution. May be `"normal"` or `"bernoulli"`.
    :param str response_label: Variable label for response.
    """
    # design is size batch x n x p
    # tau is size batch
    tau_shape = design.shape[:-2]
    if obs_sd is None:
        # First, sample tau (observation precision)
        tau_prior = dist.Gamma(alpha_0.expand(tau_shape),
                               beta_0.expand(tau_shape))
        tau = pyro.sample("tau", tau_prior)
        obs_sd = 1./torch.sqrt(tau)

    elif alpha_0 is not None or beta_0 is not None:
        warnings.warn("Values of `alpha_0` and `beta_0` unused becased"
                      "`obs_sd` was specified already.")

    # response will be shape batch x n
    obs_sd = obs_sd.expand(tau_shape).unsqueeze(-1)

    # Allow different names for different coefficient groups
    w = []
    if w_sqrtlambdas is not None:
        for name, w_sqrtlambda in w_sqrtlambdas.items():
            w_mean = w_means[name]
            # Place a normal prior on the regression coefficient
            w_prior = dist.Normal(w_mean, obs_sd / w_sqrtlambda).independent(1)
            w.append(pyro.sample(name, w_prior).unsqueeze(-1))
    w = torch.cat(w, dim=-2)

    # Run the regressor forward conditioned on inputs
    prediction_mean = torch.matmul(design, w).squeeze(-1)
    if response == "normal":
        # y is an n-vector: hence use .independent(1)
        return pyro.sample(response_label, dist.Normal(prediction_mean, obs_sd).independent(1))
    elif response == "bernoulli":
        return pyro.sample(response_label, dist.Bernoulli(logits=prediction_mean).independent(1))
    else:
        raise ValueError("Unknown response distribution: '{}'".format(response))


def normal_inv_gamma_guide(design, obs_sd, w_sizes):
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
    # design is size batch x n x p
    # tau is size batch
    tau_shape = design.shape[:-2]
    if obs_sd is None:
        # First, sample tau (observation precision)
        alpha = softplus(pyro.param("invsoftplus_alpha", 3.*torch.ones(tau_shape)))
        beta = softplus(pyro.param("invsoftplus_beta", 3.*torch.ones(tau_shape)))
        # Global variable
        tau_prior = dist.Gamma(alpha, beta)
        tau = pyro.sample("tau", tau_prior)
        obs_sd = 1./torch.sqrt(tau)

    # response will be shape batch x n
    obs_sd = obs_sd.expand(tau_shape).unsqueeze(-1)

    for name, size in w_sizes.items():
        w_shape = tau_shape + size
        # Set up mu and lambda
        mw_param = pyro.param("{}_guide_mean".format(name), torch.zeros(w_shape))
        sqrtlambda_param = softplus(pyro.param("{}_guide_sqrtlambda".format(name),
                                               3.*torch.ones(w_shape)))
        # guide distributions for w
        w_dist = dist.Normal(mw_param, obs_sd / sqrtlambda_param).independent(1)
        pyro.sample(name, w_dist)


# Some wrapper functions for common models
def zero_mean_unit_obs_sd_lm(prior_sds, intercept_sd=None):
    def model(design):
        if intercept_sd is not None:
            design = cache_constant(design)
            return bayesian_linear_model(design,
                                         w_means={"w": torch.tensor(0.),
                                                  "b": torch.tensor(0.)},
                                         w_sqrtlambdas={"w": 1./prior_sds,
                                                        "b": 1./intercept_sd.unsqueeze(-1)},
                                         obs_sd=torch.tensor(1.))
        else:
            return bayesian_linear_model(design,
                                         w_means={"w": torch.tensor(0.)},
                                         w_sqrtlambdas={"w": 1./prior_sds},
                                         obs_sd=torch.tensor(1.))

    # @lru_cache(10)
    def cache_constant(design):
        return torch.cat(design, torch.tensor(1.).expand(design.shape[:-1]+(1,)))

    def guide(design):
        if intercept_sd is not None:
            return normal_inv_gamma_guide(design,
                                          w_sizes={"w": prior_sds.shape,
                                                   "b": (1,)},
                                          obs_sd=torch.tensor(1.))
        else:
            return normal_inv_gamma_guide(design,
                                          w_sizes={"w": prior_sds.shape},
                                          obs_sd=torch.tensor(1.))

    return model, guide


def zero_mean_normal_inv_gamma(alpha, beta, prior_sds):
    def model(design):
        return bayesian_linear_model(design,
                                     w_means={"w": torch.tensor(0.)},
                                     w_sqrtlambdas={"w": 1/prior_sds},
                                     alpha_0=alpha,
                                     beta_0=beta)

    def guide(design):
        return normal_inv_gamma_guide(design,
                                      w_sizes={"w": prior_sds.shape})

    return model, guide


def two_group_bernoulli(global_sds, local_sds):
    def model(design):
        return bayesian_linear_model(design,
                                     w_means={"w_global": torch.tensor(0.),
                                              "w_local": torch.tensor(0.)},
                                     w_sqrtlambdas={"w_global": 1/global_sds,
                                                    "w_local": 1/local_sds},
                                     obs_sd=torch.tensor(1.),
                                     response="bernoulli")

    def guide(design):
        return normal_inv_gamma_guide(design, None,
                                      w_sizes={"w_global": global_sds.shape,
                                               "w_local": local_sds.shape})

    return model, guide


def group_assignment_matrix(design):
    """Converts a one-dimensional tensor listing group sizes into a
    two-dimensional binary tensor of indicator variables.

    :return: A :math:`n \times p` binary matrix where :math:`p` is
        the length of `design` and :math:`n` is its sum. There are
        :math:`n_i` ones in the :math:`i`th column.
    :rtype: torch.tensor

    """
    n, p = int(torch.sum(design)), int(design.shape[0])
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
