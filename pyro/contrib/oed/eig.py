from __future__ import absolute_import, division, print_function

import torch
import math
import warnings

import pyro
from pyro import poutine
from pyro.contrib.autoguide import mean_field_guide_entropy
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI
from pyro.util import torch_isnan, torch_isinf
from pyro.contrib.util import lexpand, rexpand


def laplace_vi_ape(model, design, observation_labels, target_labels,
                   guide, loss, optim, num_steps,
                   final_num_samples, y_dist=None):
    """
    Laplace approximation
    """
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if target_labels is not None and isinstance(target_labels, str):
        target_labels = [target_labels]

    def posterior_entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        y_dict = {label: y[i, ...] for i, label in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        # Here just using SVI to run the MAP optimization
        guide.train()
        SVI(conditioned_model, guide=guide, loss=loss, optim=optim, num_steps=num_steps, num_samples=1).run(design)
        # Recover the entropy
        with poutine.block():
            final_loss = loss(conditioned_model, guide, design)
            guide.finalize(final_loss, target_labels)
            entropy = mean_field_guide_entropy(guide, [design], whitelist=target_labels)
        return entropy

    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, num_samples=final_num_samples).run(design),
                                   sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    ape = loss_dist.mean

    return ape


# Deprecated
def vi_ape(model, design, observation_labels, target_labels,
           vi_parameters, is_parameters, y_dist=None):
    """Estimates the average posterior entropy (APE) loss function using
    variational inference (VI).

    The APE loss function estimated by this method is defined as

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
    :return: Loss function estimate
    :rtype: `torch.Tensor`

    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if target_labels is not None and isinstance(target_labels, str):
        target_labels = [target_labels]

    def posterior_entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        y_dict = {label: y[i, ...] for i, label in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        SVI(conditioned_model, **vi_parameters).run(design)
        # Recover the entropy
        with poutine.block():
            guide = vi_parameters["guide"]
            entropy = mean_field_guide_entropy(guide, [design], whitelist=target_labels)
        return entropy

    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, **is_parameters).run(design),
                                   sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    loss = loss_dist.mean

    return loss


def nmc_eig(model, design, observation_labels, target_labels=None,
            N=100, M=10, M_prime=None, independent_priors=False, N_seq=1):
    """
   Nested Monte Carlo estimate of the expected information
    gain (EIG). The estimate is, when there are not any random effects,

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N \\log p(y_n | \\theta_n, d) -
        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^M p(y_n | \\theta_m, d)\\right)

    The estimate is, in the presence of random effects,

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N  \\log \\left(\\frac{1}{M'}\\sum_{m=1}^{M'}
        p(y_n | \\theta_n, \\widetilde{\\theta}_{nm}, d)\\right)-
        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^{M}
        p(y_n | \\theta_m, \\widetilde{\\theta}_{m}, d)\\right)

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
    :return: EIG estimate
    :rtype: `torch.Tensor`
    """

    if isinstance(observation_labels, str):  # list of strings instead of strings
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    s = 0.
    if N_seq > 1:
        warnings.warn("Running nmc_eig with N_seq > 1 known to cause memory issues")
    for i in range(N_seq):
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
        s += terms.sum(0)/nonnan

    return s/N_seq


# Pre-release
def accelerated_nmc_eig(model, design, observation_labels, target_labels,
                        yspace, N=100, M_prime=None):
    """
    Unnested Monte Carlo estimate of the expected information
    gain (EIG). The estimate is, when there are not any random effects,

    .. math::

            \\frac{1}{N}\\sum_{n=1}^N\\sum_{y=1}^{|Y|}p(y | \\theta_n, d) \\log p(y | \\theta_n, d)-
            \\sum_{y=1}^{|Y|}\\left[ \\left( \\frac{1}{N} \\sum_{n=1}^N p(y | \\theta_n, d)\\right)
            log\\left(\\frac{1}{N}\\sum_{n=1}^N p(y | \\theta_n, d)\\right)\\right]

    The estimate is, in the presence of random effects,

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N\\sum_{y=1}^{|Y|}\\left(\\left(\\frac{1}{M'}\\sum_{m=1}^{M'}
        p(y | \\theta_n, \\widetilde{\\theta}_{nm}, d)\\right)
        \\log \\left(\\frac{1}{M'}\\sum_{m=1}^{M'}p(y | \\theta_n, \\widetilde{\\theta}_{nm}, d)\\right)\\right)-
        \\sum_{y=1}^{|Y|}\\left(\\left( \\frac{1}{N}\\sum_{n=1}^N p(y | \\theta_n, \\widetilde{\\theta}_{n}, d)\\right)
        \\log \\left(\\frac{1}{N}\\sum_{n=1}^N p(y | \\theta_n, \\widetilde{\\theta}_{n}, d)\\right)\\right)

    The latter form is used when `M_prime != None`.

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param dictionary yspace: maps y to a tensor that contains the possible values that y can take
    :param int N: Number of outer expectation samples.
    :param int M_prime: Number of samples for `p(y | theta, d)` if required.
    :return: EIG estimate
    :rtype: `torch.Tensor`
    """

    if isinstance(observation_labels, str):  # list of strings instead of strings
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, N, 1)  # N copies of the model
    shape = list(design.shape[:-1])
    expanded_yspace = {k: rexpand(y, *shape) for k, y in yspace.items()}
    newmodel = pyro.condition(model, data=expanded_yspace)
    trace = poutine.trace(newmodel).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)

    if M_prime is None:
        first_term = xexpx(lp).sum(0).sum(0)/N

    else:
        y_dict = {l: lexpand(trace.nodes[l]["value"], M_prime, 1) for l in observation_labels}
        theta_dict = {l: lexpand(trace.nodes[l]["value"], M_prime) for l in target_labels}
        theta_dict.update(y_dict)
        # Resample M values of theta_tilde and compute conditional probabilities
        othermodel = pyro.condition(model, data=theta_dict)
        reexpanded_design = lexpand(design, M_prime, N, 1)
        retrace = poutine.trace(othermodel).get_trace(reexpanded_design)
        retrace.compute_log_prob()
        relp = sum(retrace.nodes[l]["log_prob"] for l in observation_labels).logsumexp(0) \
            - math.log(M_prime)
        first_term = xexpx(relp).sum(0).sum(0)/N

    second_term = xexpx(lp.logsumexp(0) - math.log(N)).sum(0)
    return first_term - second_term


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
    :param int num_steps: Number of optimisation steps.
    :param function or torch.nn.Module T: optimisable function `T` for use in the
        Donsker-Varadhan loss function.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimisation.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :return: EIG estimate, optionally includes full optimisatio history
    :rtype: `torch.Tensor` or `tuple`
    """
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = donsker_varadhan_loss(model, T, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def posterior_ape(model, design, observation_labels, target_labels,
                  num_samples, num_steps, guide, optim, return_history=False,
                  final_design=None, final_num_samples=None, *args, **kwargs):
    """
    Posterior estimate of average posterior entropy (APE).

    The posterior representation of APE is

        :math:`sup_{q}E_{p(y, \\theta | d)}[\\log q(\\theta | y, d)]`

    where :math:`q` is any distribution on :math:`\\theta`.

    This method optimises the loss over a given guide family `guide`
    representing :math:`q`.

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int num_samples: Number of samples per iteration.
    :param int num_steps: Number of optimisation steps.
    :param function guide: guide family for use in the (implicit) posterior estimation.
        The parameters of `guide` are optimised to maximise the posterior
        objective.
    :param pyro.optim.Optim optim: Optimiser to use.
    :param bool return_history: If `True`, also returns a tensor giving the loss function
        at each step of the optimisation.
    :param torch.Tensor final_design: The final design tensor to evaluate at. If `None`, uses
        `design`.
    :param int final_num_samples: The number of samples to use at the final evaluation, If `None,
        uses `num_samples`.
    :return: EIG estimate, optionally includes full optimisatio history
    :rtype: `torch.Tensor` or `tuple`
    """
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = posterior_loss(model, guide, observation_labels, target_labels, *args, **kwargs)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def marginal_eig(model, design, observation_labels, target_labels,
                 num_samples, num_steps, guide, optim, return_history=False,
                 final_design=None, final_num_samples=None):
    """Estimate EIG by estimating the marginal entropy, that of :math:`p(y|d)`.

    Warning: this method does **not** estimate the correct quantity in the presence of
    random effects.
    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = marginal_loss(model, guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def marginal_likelihood_eig(model, design, observation_labels, target_labels,
                            num_samples, num_steps, marginal_guide, cond_guide, optim,
                            return_history=False, final_design=None, final_num_samples=None):
    """Estimate EIG by estimating the marginal entropy, that of :math:`p(y|d)`,
    *and* the conditional entropy, of :math:`p(y|\\theta, d)`, both via Gibbs' Inequality.
    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = marginal_likelihood_loss(model, marginal_guide, cond_guide, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


# Pre-release
def amortized_lfire_eig(model, design, observation_labels, target_labels,
                        num_samples, num_steps, classifier, optim, return_history=False,
                        final_design=None, final_num_samples=None):
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = alfire_loss(model, classifier, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples)


def lfire_eig(model, design, observation_labels, target_labels,
              num_y_samples, num_theta_samples, num_steps, classifier, optim, return_history=False,
              final_design=None, final_num_samples=None):
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = lexpand(design, num_theta_samples)
    trace = poutine.trace(model).get_trace(expanded_design)

    theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}
    cond_model = pyro.condition(model, data=theta_dict)

    loss = lfire_loss(model, cond_model, classifier, observation_labels, target_labels)
    out = opt_eig_ape_loss(expanded_design, loss, num_y_samples, num_steps, optim, return_history,
                           final_design, final_num_samples)
    if return_history:
        return out[0], out[1].sum(0) / num_theta_samples
    else:
        return out.sum(0) / num_theta_samples


def elbo_learn(model, design, observation_labels, target_labels,
               num_samples, num_steps, guide, data, optim):

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = elbo(model, guide, data, observation_labels, target_labels)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim)


def vnmc_eig(model, design, observation_labels, target_labels,
             num_samples, num_steps, guide, optim, return_history=False,
             final_design=None, final_num_samples=None):
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss = vnmc_eig_loss(model, guide, observation_labels, target_labels)
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

    _, loss = loss_fn(final_design, final_num_samples, evaluation=True)
    if return_history:
        return torch.stack(history), loss
    else:
        return loss


def donsker_varadhan_loss(model, T, observation_labels, target_labels):

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


def posterior_loss(model, guide, observation_labels, target_labels, analytic_entropy=False):

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
            loss = mean_field_guide_entropy(
                guide, [y_dict, expanded_design, observation_labels, target_labels],
                whitelist=target_labels).sum(0)/num_particles
            agg_loss = loss.sum()
        else:
            terms = -sum(cond_trace.nodes[l]["log_prob"] for l in target_labels)
            agg_loss, loss = safe_mean_terms(terms)

        return agg_loss, loss

    return loss_fn


def marginal_loss(model, guide, observation_labels, target_labels):

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

        return safe_mean_terms(terms)

    return loss_fn


def marginal_likelihood_loss(model, marginal_guide, likelihood_guide, observation_labels, target_labels):

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

        return safe_mean_terms(terms)

    return loss_fn


def alfire_loss(model, h, observation_labels, target_labels):

    def loss_fn(design, num_particles, evaluation=False, **kwargs):

        try:
            pyro.module("h", h)
        except AssertionError:
            pass

        expanded_design = lexpand(design, num_particles)

        # Unshuffled data
        unshuffled_trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: unshuffled_trace.nodes[l]["value"] for l in observation_labels}

        if not evaluation:
            # Shuffled data
            # Not actually shuffling, re-simulate for safety
            conditional_model = pyro.condition(model, data=y_dict)
            shuffled_trace = poutine.trace(conditional_model).get_trace(expanded_design)

            h_joint = h(expanded_design, unshuffled_trace, observation_labels, target_labels)
            h_independent = h(expanded_design, shuffled_trace, observation_labels, target_labels)

            terms = torch.nn.functional.softplus(-h_joint) + torch.nn.functional.softplus(h_independent)
            return safe_mean_terms(terms)

        else:
            h_joint = h(expanded_design, unshuffled_trace, observation_labels, target_labels)
            return safe_mean_terms(h_joint)

    return loss_fn


def lfire_loss(model_marginal, model_conditional, h, observation_labels, target_labels):

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
            return safe_mean_terms(terms)

        else:
            h_joint = h(expanded_design, model_conditional_trace, observation_labels, target_labels)
            return safe_mean_terms(h_joint)

    return loss_fn


def elbo(model, guide, data, observation_labels, target_labels):

    def loss_fn(design, num_particles, **kwargs):

        y_dict = {l: lexpand(y, num_particles) for l, y in data.items()}

        expanded_design = lexpand(design, num_particles)

        # Sample from q(theta)
        trace = poutine.trace(guide).get_trace(expanded_design)
        theta_y_dict = {l: trace.nodes[l]["value"] for l in target_labels}
        theta_y_dict.update(y_dict)
        trace.compute_log_prob()

        # Run through p(theta)
        modelp = pyro.condition(model, data=theta_y_dict)
        model_trace = poutine.trace(modelp).get_trace(expanded_design)
        model_trace.compute_log_prob()

        terms = sum(trace.nodes[l]["log_prob"] for l in target_labels)
        terms -= sum(model_trace.nodes[l]["log_prob"] for l in target_labels)
        terms -= sum(model_trace.nodes[l]["log_prob"] for l in observation_labels)

        return safe_mean_terms(terms)

    return loss_fn


def vnmc_eig_loss(model, guide, observation_labels, target_labels):

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

        return safe_mean_terms(terms)

    return loss_fn


def safe_mean_terms(terms):
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

    This function makes the outputs more stable when the inputs of this function converge to -infinity

    Args:
        a: torch.Tensor

    Returns:
        Equivalent of `a*torch.exp(a)`.
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


class EwmaLog(object):
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
