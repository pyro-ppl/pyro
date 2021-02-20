# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import math
import warnings

import pyro
from pyro import poutine
from pyro.infer.autoguide.utils import mean_field_entropy
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI
from pyro.util import torch_isnan, torch_isinf
from pyro.contrib.util import lexpand

__all__ = [
    "laplace_eig",
    "vi_eig",
    "nmc_eig",
    "donsker_varadhan_eig",
    "posterior_eig",
    "marginal_eig",
    "lfire_eig",
    "vnmc_eig"
]


def laplace_eig(model, design, observation_labels, target_labels, guide, loss, optim, num_steps,
                final_num_samples, y_dist=None, eig=True, **prior_entropy_kwargs):
    """
    Estimates the expected information gain (EIG) by making repeated Laplace approximations to the posterior.

    :param function model: Pyro stochastic function taking `design` as only argument.
    :param torch.Tensor design: Tensor of possible designs.
    :param list observation_labels: labels of sample sites to be regarded as observables.
    :param list target_labels: labels of sample sites to be regarded as latent variables of interest, i.e. the sites
        that we wish to gain information about.
    :param function guide: Pyro stochastic function corresponding to `model`.
    :param loss: a Pyro loss such as `pyro.infer.Trace_ELBO().differentiable_loss`.
    :param optim: optimizer for the loss
    :param int num_steps: Number of gradient steps to take per sampled pseudo-observation.
    :param int final_num_samples: Number of `y` samples (pseudo-observations) to take.
    :param y_dist: Distribution to sample `y` from- if `None` we use the Bayesian marginal distribution.
    :param bool eig: Whether to compute the EIG or the average posterior entropy (APE). The EIG is given by
        `EIG = prior entropy - APE`. If `True`, the prior entropy will be estimated analytically,
        or by Monte Carlo as appropriate for the `model`. If `False` the APE is returned.
    :param dict prior_entropy_kwargs: parameters for estimating the prior entropy: `num_prior_samples` indicating the
        number of samples for a MC estimate of prior entropy, and `mean_field` indicating if an analytic form for
        a mean-field prior should be tried.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor
    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if target_labels is not None and isinstance(target_labels, str):
        target_labels = [target_labels]

    ape = _laplace_vi_ape(model, design, observation_labels, target_labels, guide, loss, optim, num_steps,
                          final_num_samples, y_dist=y_dist)
    return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)


def _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs):
    mean_field = prior_entropy_kwargs.get("mean_field", True)
    if eig:
        if mean_field:
            try:
                prior_entropy = mean_field_entropy(model, [design], whitelist=target_labels)
            except NotImplemented:
                prior_entropy = monte_carlo_entropy(model, design, target_labels, **prior_entropy_kwargs)
        else:
            prior_entropy = monte_carlo_entropy(model, design, target_labels, **prior_entropy_kwargs)
        return prior_entropy - ape
    else:
        return ape


def _laplace_vi_ape(model, design, observation_labels, target_labels, guide, loss, optim, num_steps,
                    final_num_samples, y_dist=None):

    def posterior_entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        y_dict = {label: y[i, ...] for i, label in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        # Here just using SVI to run the MAP optimization
        guide.train()
        svi = SVI(conditioned_model, guide=guide, loss=loss, optim=optim)
        with poutine.block():
            for _ in range(num_steps):
                svi.step(design)
        # Recover the entropy
        with poutine.block():
            final_loss = loss(conditioned_model, guide, design)
            guide.finalize(final_loss, target_labels)
            entropy = mean_field_entropy(guide, [design], whitelist=target_labels)
        return entropy

    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, num_samples=final_num_samples).run(design),
                                   sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    ape = loss_dist.mean

    return ape


# Deprecated
def vi_eig(model, design, observation_labels, target_labels, vi_parameters, is_parameters, y_dist=None,
           eig=True, **prior_entropy_kwargs):
    """.. deprecated:: 0.4.1
        Use `posterior_eig` instead.

    Estimates the expected information gain (EIG) using variational inference (VI).

    The APE is defined as

        :math:`APE(d)=E_{Y\\sim p(y|\\theta, d)}[H(p(\\theta|Y, d))]`

    where :math:`H[p(x)]` is the `differential entropy
    <https://en.wikipedia.org/wiki/Differential_entropy>`_.
    The APE is related to expected information gain (EIG) by the equation

        :math:`EIG(d)=H[p(\\theta)]-APE(d)`

    in particular, minimising the APE is equivalent to maximising EIG.

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param dict vi_parameters: Variational inference parameters which should include:
        `optim`: an instance of :class:`pyro.Optim`, `guide`: a guide function
        compatible with `model`, `num_steps`: the number of VI steps to make,
        and `loss`: the loss function to use for VI
    :param dict is_parameters: Importance sampling parameters for the
        marginal distribution of :math:`Y`. May include `num_samples`: the number
        of samples to draw from the marginal.
    :param pyro.distributions.Distribution y_dist: (optional) the distribution
        assumed for the response variable :math:`Y`
    :param bool eig: Whether to compute the EIG or the average posterior entropy (APE). The EIG is given by
        `EIG = prior entropy - APE`. If `True`, the prior entropy will be estimated analytically,
        or by Monte Carlo as appropriate for the `model`. If `False` the APE is returned.
    :param dict prior_entropy_kwargs: parameters for estimating the prior entropy: `num_prior_samples` indicating the
        number of samples for a MC estimate of prior entropy, and `mean_field` indicating if an analytic form for
        a mean-field prior should be tried.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor

    """

    warnings.warn("`vi_eig` is deprecated in favour of the amortized version: `posterior_eig`.", DeprecationWarning)

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if target_labels is not None and isinstance(target_labels, str):
        target_labels = [target_labels]

    ape = _vi_ape(model, design, observation_labels, target_labels, vi_parameters, is_parameters, y_dist=y_dist)
    return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)


def _vi_ape(model, design, observation_labels, target_labels, vi_parameters, is_parameters, y_dist=None):
    svi_num_steps = vi_parameters.pop('num_steps')

    def posterior_entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        y_dict = {label: y[i, ...] for i, label in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        svi = SVI(conditioned_model, **vi_parameters)
        with poutine.block():
            for _ in range(svi_num_steps):
                svi.step(design)
        # Recover the entropy
        with poutine.block():
            guide = vi_parameters["guide"]
            entropy = mean_field_entropy(guide, [design], whitelist=target_labels)
        return entropy

    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, **is_parameters).run(design),
                                   sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    loss = loss_dist.mean

    return loss


def nmc_eig(model, design, observation_labels, target_labels=None,
            N=100, M=10, M_prime=None, independent_priors=False):
    """Nested Monte Carlo estimate of the expected information
    gain (EIG). The estimate is, when there are not any random effects,

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N \\log p(y_n | \\theta_n, d) -
        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^M p(y_n | \\theta_m, d)\\right)

    where :math:`\\theta_n, y_n \\sim p(\\theta, y | d)` and :math:`\\theta_m \\sim p(\\theta)`.
    The estimate in the presence of random effects is

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N  \\log \\left(\\frac{1}{M'}\\sum_{m=1}^{M'}
        p(y_n | \\theta_n, \\widetilde{\\theta}_{nm}, d)\\right)-
        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^{M}
        p(y_n | \\theta_m, \\widetilde{\\theta}_{m}, d)\\right)

    where :math:`\\widetilde{\\theta}` are the random effects with
    :math:`\\widetilde{\\theta}_{nm} \\sim p(\\widetilde{\\theta}|\\theta=\\theta_n)` and
    :math:`\\theta_m,\\widetilde{\\theta}_m \\sim p(\\theta,\\widetilde{\\theta})`.
    The latter form is used when `M_prime != None`.

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int N: Number of outer expectation samples.
    :param int M: Number of inner expectation samples for `p(y|d)`.
    :param int M_prime: Number of samples for `p(y | theta, d)` if required.
    :param bool independent_priors: Only used when `M_prime` is not `None`. Indicates whether the prior distributions
        for the target variables and the nuisance variables are independent. In this case, it is not necessary to
        sample the targets conditional on the nuisance variables.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor
    """

    if isinstance(observation_labels, str):  # list of strings instead of strings
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    trace = poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()

    if M_prime is not None:
        y_dict = {l: lexpand(trace.nodes[l]["value"], M_prime) for l in observation_labels}
        theta_dict = {l: lexpand(trace.nodes[l]["value"], M_prime) for l in target_labels}
        theta_dict.update(y_dict)
        # Resample M values of u and compute conditional probabilities
        # WARNING: currently the use of condition does not actually sample
        # the conditional distribution!
        # We need to use some importance weighting
        conditional_model = pyro.condition(model, data=theta_dict)
        if independent_priors:
            reexpanded_design = lexpand(design, M_prime, 1)
        else:
            # Not acceptable to use (M_prime, 1) here - other variables may occur after
            # theta, so need to be sampled conditional upon it
            reexpanded_design = lexpand(design, M_prime, N)
        retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
        retrace.compute_log_prob()
        conditional_lp = sum(retrace.nodes[l]["log_prob"] for l in observation_labels).logsumexp(0) \
            - math.log(M_prime)
    else:
        # This assumes that y are independent conditional on theta
        # Furthermore assume that there are no other variables besides theta
        conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)

    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, 1)  # sample M theta
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    marginal_lp = sum(retrace.nodes[l]["log_prob"] for l in observation_labels).logsumexp(0) \
        - math.log(M)

    terms = conditional_lp - marginal_lp
    nonnan = (~torch.isnan(terms)).sum(0).type_as(terms)
    terms[torch.isnan(terms)] = 0.
    return terms.sum(0)/nonnan


def donsker_varadhan_eig(model, design, observation_labels, target_labels,
                         num_samples, num_steps, T, optim, return_history=False,
                         final_design=None, final_num_samples=None):
    """
    Donsker-Varadhan estimate of the expected information gain (EIG).

    The Donsker-Varadhan representation of EIG is

    .. math::

        \\sup_T E_{p(y, \\theta | d)}[T(y, \\theta)] - \\log E_{p(y|d)p(\\theta)}[\\exp(T(\\bar{y}, \\bar{\\theta}))]

    where :math:`T` is any (measurable) function.

    This methods optimises the loss function over a pre-specified class of
    functions `T`.

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int num_samples: Number of samples per iteration.
    :param int num_steps: Number of optimization steps.
    :param function or torch.nn.Module T: optimisable function `T` for use in the
        Donsker-Varadhan loss function.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimization.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor or tuple
    """
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _donsker_varadhan_loss(model, T, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def posterior_eig(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim,
                  return_history=False, final_design=None, final_num_samples=None, eig=True, prior_entropy_kwargs={},
                  *args, **kwargs):
    """
    Posterior estimate of expected information gain (EIG) computed from the average posterior entropy (APE)
    using :math:`EIG(d) = H[p(\\theta)] - APE(d)`. See [1] for full details.

    The posterior representation of APE is

        :math:`\\sup_{q}\\ E_{p(y, \\theta | d)}[\\log q(\\theta | y, d)]`

    where :math:`q` is any distribution on :math:`\\theta`.

    This method optimises the loss over a given `guide` family representing :math:`q`.

    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int num_samples: Number of samples per iteration.
    :param int num_steps: Number of optimization steps.
    :param function guide: guide family for use in the (implicit) posterior estimation.
        The parameters of `guide` are optimised to maximise the posterior
        objective.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimization.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :param bool eig: Whether to compute the EIG or the average posterior entropy (APE). The EIG is given by
        `EIG = prior entropy - APE`. If `True`, the prior entropy will be estimated analytically,
        or by Monte Carlo as appropriate for the `model`. If `False` the APE is returned.
    :param dict prior_entropy_kwargs: parameters for estimating the prior entropy: `num_prior_samples` indicating the
        number of samples for a MC estimate of prior entropy, and `mean_field` indicating if an analytic form for
        a mean-field prior should be tried.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor or tuple
    """
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    ape = _posterior_ape(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim,
                         return_history=return_history, final_design=final_design, final_num_samples=final_num_samples,
                         *args, **kwargs)
    return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)


def _posterior_ape(model, design, observation_labels, target_labels,
                   num_samples, num_steps, guide, optim, return_history=False,
                   final_design=None, final_num_samples=None, *args, **kwargs):

    loss = _posterior_loss(model, guide, observation_labels, target_labels, *args, **kwargs)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def marginal_eig(model, design, observation_labels, target_labels,
                 num_samples, num_steps, guide, optim, return_history=False,
                 final_design=None, final_num_samples=None):
    """Estimate EIG by estimating the marginal entropy :math:`p(y|d)`. See [1] for full details.

    The marginal representation of EIG is

        :math:`\\inf_{q}\\ E_{p(y, \\theta | d)}\\left[\\log \\frac{p(y | \\theta, d)}{q(y | d)} \\right]`

    where :math:`q` is any distribution on :math:`y`. A variational family for :math:`q` is specified in the `guide`.

    .. warning :: This method does **not** estimate the correct quantity in the presence of random effects.

    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int num_samples: Number of samples per iteration.
    :param int num_steps: Number of optimization steps.
    :param function guide: guide family for use in the marginal estimation.
        The parameters of `guide` are optimised to maximise the log-likelihood objective.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimization.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor or tuple
    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _marginal_loss(model, guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def marginal_likelihood_eig(model, design, observation_labels, target_labels,
                            num_samples, num_steps, marginal_guide, cond_guide, optim,
                            return_history=False, final_design=None, final_num_samples=None):
    """Estimates EIG by estimating the marginal entropy, that of :math:`p(y|d)`,
    *and* the conditional entropy, of :math:`p(y|\\theta, d)`, both via Gibbs' Inequality. See [1] for full details.

    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int num_samples: Number of samples per iteration.
    :param int num_steps: Number of optimization steps.
    :param function marginal_guide: guide family for use in the marginal estimation.
        The parameters of `guide` are optimised to maximise the log-likelihood objective.
    :param function cond_guide: guide family for use in the likelihood (conditional) estimation.
        The parameters of `guide` are optimised to maximise the log-likelihood objective.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimization.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor or tuple
    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _marginal_likelihood_loss(model, marginal_guide, cond_guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def lfire_eig(model, design, observation_labels, target_labels,
              num_y_samples, num_theta_samples, num_steps, classifier, optim, return_history=False,
              final_design=None, final_num_samples=None):
    """Estimates the EIG using the method of Likelihood-Free Inference by Ratio Estimation (LFIRE) as in [1].
    LFIRE is run separately for several samples of :math:`\\theta`.

    [1] Kleinegesse, Steven, and Michael Gutmann. "Efficient Bayesian Experimental Design for Implicit Models."
    arXiv preprint arXiv:1810.09912 (2018).

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int num_y_samples: Number of samples to take in :math:`y` for each :math:`\\theta`.
    :param: int num_theta_samples: Number of initial samples in :math:`\\theta` to take. The likelihood ratio
                                   is estimated by LFIRE for each sample.
    :param int num_steps: Number of optimization steps.
    :param function classifier: a Pytorch or Pyro classifier used to distinguish between samples of :math:`y` under
                                :math:`p(y|d)` and samples under :math:`p(y|\\theta,d)` for some :math:`\\theta`.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimization.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor or tuple
    """
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = lexpand(design, num_theta_samples)
    trace = poutine.trace(model).get_trace(expanded_design)

    theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
    cond_model = pyro.condition(model, data=theta_dict)

    loss = _lfire_loss(model, cond_model, classifier, observation_labels, target_labels)
    out = opt_eig_ape_loss(expanded_design, loss, num_y_samples, num_steps, optim, return_history,
                           final_design, final_num_samples)
    if return_history:
        return out[0], out[1].sum(0) / num_theta_samples
    else:
        return out.sum(0) / num_theta_samples


def vnmc_eig(model, design, observation_labels, target_labels,
             num_samples, num_steps, guide, optim, return_history=False,
             final_design=None, final_num_samples=None):
    """Estimates the EIG using Variational Nested Monte Carlo (VNMC). The VNMC estimate [1] is

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N \\left[ \\log p(y_n | \\theta_n, d) -
         \\log \\left(\\frac{1}{M}\\sum_{m=1}^M \\frac{p(\\theta_{mn})p(y_n | \\theta_{mn}, d)}
         {q(\\theta_{mn} | y_n)} \\right) \\right]

    where :math:`q(\\theta | y)` is the learned variational posterior approximation and
    :math:`\\theta_n, y_n \\sim p(\\theta, y | d)`, :math:`\\theta_{mn} \\sim q(\\theta|y=y_n)`.

    As :math:`N \\to \\infty` this is an upper bound on EIG. We minimise this upper bound by stochastic gradient
    descent.

    .. warning :: This method cannot be used in the presence of random effects.

    [1] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param tuple num_samples: Number of (:math:`N, M`) samples per iteration.
    :param int num_steps: Number of optimization steps.
    :param function guide: guide family for use in the posterior estimation.
        The parameters of `guide` are optimised to minimise the VNMC upper bound.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimization.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param tuple final_num_samples: The number of (:math:`N, M`) samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor or tuple
    """
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = _vnmc_eig_loss(model, guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def opt_eig_ape_loss(design, loss_fn, num_samples, num_steps, optim, return_history=False,
                     final_design=None, final_num_samples=None):

    if final_design is None:
        final_design = design
    if final_num_samples is None:
        final_num_samples = num_samples

    params = None
    history = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, evaluation=return_history)
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward(retain_graph=True)
        if return_history:
            history.append(loss)
        optim(params)
        try:
            optim.step()
        except AttributeError:
            pass

    _, loss = loss_fn(final_design, final_num_samples, evaluation=True)
    if return_history:
        return torch.stack(history), loss
    else:
        return loss


def monte_carlo_entropy(model, design, target_labels, num_prior_samples=1000):
    """Computes a Monte Carlo estimate of the entropy of `model` assuming that each of sites in `target_labels` is
    independent and the entropy is to be computed for that subset of sites only.
    """

    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_prior_samples)
    trace = pyro.poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    return -lp.sum(0) / num_prior_samples


def _donsker_varadhan_loss(model, T, observation_labels, target_labels):
    """DV loss: to evaluate directly use `donsker_varadhan_eig` setting `num_steps=0`."""

    ewma_log = EwmaLog(alpha=0.90)

    def loss_fn(design, num_particles, **kwargs):

        try:
            pyro.module("T", T)
        except AssertionError:
            pass

        expanded_design = lexpand(design, num_particles)

        # Unshuffled data
        unshuffled_trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: unshuffled_trace.nodes[l]["value"] for l in observation_labels}

        # Shuffled data
        # Not actually shuffling, resimulate for safety
        conditional_model = pyro.condition(model, data=y_dict)
        shuffled_trace = poutine.trace(conditional_model).get_trace(expanded_design)

        T_joint = T(expanded_design, unshuffled_trace, observation_labels, target_labels)
        T_independent = T(expanded_design, shuffled_trace, observation_labels, target_labels)

        joint_expectation = T_joint.sum(0)/num_particles

        A = T_independent - math.log(num_particles)
        s, _ = torch.max(A, dim=0)
        independent_expectation = s + ewma_log((A - s).exp().sum(dim=0), s)

        loss = joint_expectation - independent_expectation
        # Switch sign, sum over batch dimensions for scalar loss
        agg_loss = -loss.sum()
        return agg_loss, loss

    return loss_fn


def _posterior_loss(model, guide, observation_labels, target_labels, analytic_entropy=False):
    """Posterior loss: to evaluate directly use `posterior_eig` setting `num_steps=0`, `eig=False`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):

        expanded_design = lexpand(design, num_particles)

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        # Run through q(theta | y, d)
        conditional_guide = pyro.condition(guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()
        if evaluation and analytic_entropy:
            loss = mean_field_entropy(
                guide, [y_dict, expanded_design, observation_labels, target_labels],
                whitelist=target_labels).sum(0) / num_particles
            agg_loss = loss.sum()
        else:
            terms = -sum(cond_trace.nodes[l]["log_prob"] for l in target_labels)
            agg_loss, loss = _safe_mean_terms(terms)

        return agg_loss, loss

    return loss_fn


def _marginal_loss(model, guide, observation_labels, target_labels):
    """Marginal loss: to evaluate directly use `marginal_eig` setting `num_steps=0`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):

        expanded_design = lexpand(design, num_particles)

        # Sample from p(y | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}

        # Run through q(y | d)
        conditional_guide = pyro.condition(guide, data=y_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
             expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        terms = -sum(cond_trace.nodes[l]["log_prob"] for l in observation_labels)

        # At eval time, add p(y | theta, d) terms
        if evaluation:
            trace.compute_log_prob()
            terms += sum(trace.nodes[l]["log_prob"] for l in observation_labels)

        return _safe_mean_terms(terms)

    return loss_fn


def _marginal_likelihood_loss(model, marginal_guide, likelihood_guide, observation_labels, target_labels):
    """Marginal_likelihood loss: to evaluate directly use `marginal_likelihood_eig` setting `num_steps=0`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):

        expanded_design = lexpand(design, num_particles)

        # Sample from p(y | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        # Run through q(y | d)
        qyd = pyro.condition(marginal_guide, data=y_dict)
        marginal_trace = poutine.trace(qyd).get_trace(
             expanded_design, observation_labels, target_labels)
        marginal_trace.compute_log_prob()

        # Run through q(y | theta, d)
        qythetad = pyro.condition(likelihood_guide, data=y_dict)
        cond_trace = poutine.trace(qythetad).get_trace(
                theta_dict, expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()
        terms = -sum(marginal_trace.nodes[l]["log_prob"] for l in observation_labels)

        # At evaluation time, use the right estimator, q(y | theta, d) - q(y | d)
        # At training time, use -q(y | theta, d) - q(y | d) so gradients go the same way
        if evaluation:
            terms += sum(cond_trace.nodes[l]["log_prob"] for l in observation_labels)
        else:
            terms -= sum(cond_trace.nodes[l]["log_prob"] for l in observation_labels)

        return _safe_mean_terms(terms)

    return loss_fn


def _lfire_loss(model_marginal, model_conditional, h, observation_labels, target_labels):
    """LFIRE loss: to evaluate directly use `lfire_eig` setting `num_steps=0`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):

        try:
            pyro.module("h", h)
        except AssertionError:
            pass

        expanded_design = lexpand(design, num_particles)
        model_conditional_trace = poutine.trace(model_conditional).get_trace(expanded_design)

        if not evaluation:
            model_marginal_trace = poutine.trace(model_marginal).get_trace(expanded_design)

            h_joint = h(expanded_design, model_conditional_trace, observation_labels, target_labels)
            h_independent = h(expanded_design, model_marginal_trace, observation_labels, target_labels)

            terms = torch.nn.functional.softplus(-h_joint) + torch.nn.functional.softplus(h_independent)
            return _safe_mean_terms(terms)

        else:
            h_joint = h(expanded_design, model_conditional_trace, observation_labels, target_labels)
            return _safe_mean_terms(h_joint)

    return loss_fn


def _vnmc_eig_loss(model, guide, observation_labels, target_labels):
    """VNMC loss: to evaluate directly use `vnmc_eig` setting `num_steps=0`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        N, M = num_particles
        expanded_design = lexpand(design, N)

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}

        # Sample M times from q(theta | y, d) for each y
        reexpanded_design = lexpand(expanded_design, M)
        conditional_guide = pyro.condition(guide, data=y_dict)
        guide_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, reexpanded_design, observation_labels, target_labels)
        theta_y_dict = {l: guide_trace.nodes[l]["value"] for l in target_labels}
        theta_y_dict.update(y_dict)
        guide_trace.compute_log_prob()

        # Re-run that through the model to compute the joint
        modelp = pyro.condition(model, data=theta_y_dict)
        model_trace = poutine.trace(modelp).get_trace(reexpanded_design)
        model_trace.compute_log_prob()

        terms = -sum(guide_trace.nodes[l]["log_prob"] for l in target_labels)
        terms += sum(model_trace.nodes[l]["log_prob"] for l in target_labels)
        terms += sum(model_trace.nodes[l]["log_prob"] for l in observation_labels)
        terms = -terms.logsumexp(0) + math.log(M)

        # At eval time, add p(y | theta, d) terms
        if evaluation:
            trace.compute_log_prob()
            terms += sum(trace.nodes[l]["log_prob"] for l in observation_labels)

        return _safe_mean_terms(terms)

    return loss_fn


def _safe_mean_terms(terms):
    mask = torch.isnan(terms) | (terms == float('-inf')) | (terms == float('inf'))
    if terms.dtype is torch.float32:
        nonnan = (~mask).sum(0).float()
    elif terms.dtype is torch.float64:
        nonnan = (~mask).sum(0).double()
    terms[mask] = 0.
    loss = terms.sum(0) / nonnan
    agg_loss = loss.sum()
    return agg_loss, loss


def xexpx(a):
    """Computes `a*exp(a)`.

    This function makes the outputs more stable when the inputs of this function converge to :math:`-\\infty`.

    :param torch.Tensor a:
    :return: Equivalent of `a*torch.exp(a)`.
    """
    mask = (a == float('-inf'))
    y = a*torch.exp(a)
    y[mask] = 0.
    return y


class _EwmaLogFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ewma):
        ctx.save_for_backward(ewma)
        return input.log()

    @staticmethod
    def backward(ctx, grad_output):
        ewma, = ctx.saved_tensors
        return grad_output / ewma, None


_ewma_log_fn = _EwmaLogFn.apply


class EwmaLog:
    """Logarithm function with exponentially weighted moving average
    for gradients.

    For input `inputs` this function return :code:`inputs.log()`. However, it
    computes the gradient as

        :math:`\\frac{\\sum_{t=0}^{T-1} \\alpha^t}{\\sum_{t=0}^{T-1} \\alpha^t x_{T-t}}`

    where :math:`x_t` are historical input values passed to this function,
    :math:`x_T` being the most recently seen value.

    This gradient may help with numerical stability when the sequence of
    inputs to the function form a convergent sequence.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.ewma = 0.
        self.n = 0
        self.s = 0.

    def __call__(self, inputs, s, dim=0, keepdim=False):
        """Updates the moving average, and returns :code:`inputs.log()`.
        """
        self.n += 1
        if torch_isnan(self.ewma) or torch_isinf(self.ewma):
            ewma = inputs
        else:
            ewma = inputs * (1. - self.alpha) / (1 - self.alpha**self.n) \
                    + torch.exp(self.s - s) * self.ewma \
                    * (self.alpha - self.alpha**self.n) / (1 - self.alpha**self.n)
        self.ewma = ewma.detach()
        self.s = s.detach()
        return _ewma_log_fn(inputs, ewma)
