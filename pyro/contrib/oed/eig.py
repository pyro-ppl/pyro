from __future__ import absolute_import, division, print_function

import torch
import numpy as np

import pyro
from pyro import poutine
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI
from pyro.contrib.autoguide import mean_field_guide_entropy
from pyro.contrib.util import lexpand


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
        return mean_field_guide_entropy(vi_parameters["guide"], [design], whitelist=target_labels)

    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, **is_parameters).run(design),
                                   sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    loss = loss_dist.mean

    return loss


def naive_rainforth_eig(model, design, observation_labels, target_labels=None,
                        N=100, M=10, M_prime=None):
    """
    Naive Rainforth (i.e. Nested Monte Carlo) estimate of the expected information
    gain (EIG). The estimate is

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N \\log p(y_n | \\theta_n, d) -
        \\log \\left(\\frac{1}{M}\\sum_{m=1}^M p(y_n | \\theta_m, d)\\right)

    Monte Carlo estimation is attempted for the :math:`\\log p(y | \\theta, d)` term if
    the parameter `M_prime` is passed. Otherwise, it is assumed that that :math:`\\log p(y | \\theta, d)`
    can safely be read from the model itself.

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

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = lexpand(design, N)
    trace = poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()

    if M_prime is not None:
        y_dict = {l: lexpand(trace.nodes[l]["value"], M_prime) for l in observation_labels}
        theta_dict = {l: lexpand(trace.nodes[l]["value"], M_prime) for l in target_labels}
        theta_dict.update(y_dict)
        # Resample M values of u and compute conditional probabilities
        conditional_model = pyro.condition(model, data=theta_dict)
        # Not acceptable to use (M_prime, 1) here - other variables may occur after
        # theta, so need to be sampled conditional upon it
        reexpanded_design = lexpand(design, M_prime, N)
        retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
        retrace.compute_log_prob()
        conditional_lp = logsumexp(sum(retrace.nodes[l]["log_prob"] for l in observation_labels), 0) \
            - np.log(M_prime)
    else:
        # This assumes that y are independent conditional on theta
        # Furthermore assume that there are no other variables besides theta
        conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)

    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, 1)
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    marginal_lp = logsumexp(sum(retrace.nodes[l]["log_prob"] for l in observation_labels), 0) \
        - np.log(M)

    return (conditional_lp - marginal_lp).sum(0)/N


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


def barber_agakov_ape(model, design, observation_labels, target_labels,
                      num_samples, num_steps, guide, optim, return_history=False,
                      final_design=None, final_num_samples=None):
    """
    Barber-Agakov estimate of average posterior entropy (APE).

    The Barber-Agakov representation of APE is

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
        The parameters of `guide` are optimised to maximise the Barber-Agakov
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
    loss = barber_agakov_loss(model, guide, observation_labels, target_labels)
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
        agg_loss, loss = loss_fn(design, num_samples)
        agg_loss.backward()
        if return_history:
            history.append(loss)
        params = [pyro.param(name).unconstrained()
                  for name in pyro.get_param_store().get_all_param_names()]
        optim(params)
    _, loss = loss_fn(final_design, final_num_samples)
    if return_history:
        return torch.stack(history), loss
    else:
        return loss


def donsker_varadhan_loss(model, T, observation_labels, target_labels):

    ewma_log = EwmaLog(alpha=0.90)

    try:
        pyro.module("T", T)
    except AssertionError:
        pass

    def loss_fn(design, num_particles):

        expanded_design = lexpand(design, num_particles)

        # Unshuffled data
        unshuffled_trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: unshuffled_trace.nodes[l]["value"] for l in observation_labels}

        # Shuffled data
        # Not actually shuffling, resimulate for safety
        conditional_model = pyro.condition(model, data=y_dict)
        shuffled_trace = poutine.trace(conditional_model).get_trace(expanded_design)

        T_joint = T(expanded_design, unshuffled_trace, observation_labels,
                    target_labels)
        T_independent = T(expanded_design, shuffled_trace, observation_labels,
                          target_labels)

        joint_expectation = T_joint.sum(0)/num_particles

        A = T_independent - np.log(num_particles)
        s, _ = torch.max(A, dim=0)
        independent_expectation = s + ewma_log((A - s).exp().sum(dim=0), s)

        loss = joint_expectation - independent_expectation
        # Switch sign, sum over batch dimensions for scalar loss
        agg_loss = -loss.sum()
        return agg_loss, loss

    return loss_fn


def barber_agakov_loss(model, guide, observation_labels, target_labels):

    def loss_fn(design, num_particles):

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

        loss = -sum(cond_trace.nodes[l]["log_prob"] for l in target_labels).sum(0)/num_particles
        agg_loss = loss.sum()
        return agg_loss, loss

    return loss_fn


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of `log(sum(exp(inputs), dim=dim, keepdim=keepdim))`.
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class EwmaLog(torch.autograd.Function):
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
        self.ewma = torch.tensor(0.)
        self.n = 0
        self.s = 0.

    def forward(self, inputs, s, dim=0, keepdim=False):
        """Updates the moving average, and returns :code:`inputs.log()`.
        """
        self.n += 1
        if torch.isnan(self.ewma).any() or (self.ewma == float('inf')).any():
            self.ewma = inputs
            self.s = s
        else:
            self.ewma = inputs*(1. - self.alpha)/(1 - self.alpha**self.n) \
                        + torch.exp(self.s - s)*self.ewma \
                        * (self.alpha - self.alpha**self.n)/(1 - self.alpha**self.n)
            self.s = s
        return inputs.log()

    def backward(self, grad_output):
        """Returns the gradient from exponentially weighted moving
        average of historical input values.
        """
        return grad_output/self.ewma, None, None, None
