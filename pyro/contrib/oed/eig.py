import torch
import numpy as np

import pyro
from pyro import poutine
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI
from pyro.contrib.autoguide import mean_field_guide_entropy


def vi_ape(model, design, observation_labels, vi_parameters, is_parameters,
           y_dist=None, target_labels=None):
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
    :param dict vi_parameters: Variational inference parameters which should include:
        `optim`: an instance of :class:`pyro.Optim`, `guide`: a guide function
        compatible with `model`, `num_steps`: the number of VI steps to make,
        and `loss`: the loss function to use for VI
    :param dict is_parameters: Importance sampling parameters for the
        marginal distribution of :math:`Y`. May include `num_samples`: the number
        of samples to draw from the marginal.
    :param pyro.distributions.Distribution y_dist: (optional) the distribution
        assumed for the response variable :math:`Y`
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured. If `None` is passed, the posterior over all
        non-observation sites is included in the APE.
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
        return mean_field_guide_entropy(vi_parameters["guide"], design, whitelist=target_labels)

    if y_dist is None:
        y_dist = EmpiricalMarginal(Importance(model, **is_parameters).run(design),
                                   sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    loss = loss_dist.mean

    return loss


def naive_rainforth(model, design, observation_label="y", target_labels="theta",
                    N=100, M=10):
    """
    Naive Rainforth (i.e. Nested Monte Carlo) estimate of the expected information
    gain (EIG). The estimate is

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N \\log p(y_n | \\theta_n, d) - \\log (\\frac{1}{M}\\sum_{m=1}^M p(y_n | \\theta_m, d))

    Caution: the target labels must encompass all other variables in the model: no
    Monte Carlo estimation is attempted for the :math:`\\log p(y | \\theta, d)` term.
    """

    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = design.expand((N,) + design.shape)
    trace = poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    y = trace.nodes[observation_label]["value"]
    conditional_lp = trace.nodes[observation_label]["log_prob"]

    # Take M independent samples of theta
    reexpanded_design = design.expand((M, 1) + design.shape)
    reexp_trace = poutine.trace(model).get_trace(reexpanded_design)
    data = {l: reexp_trace.nodes[l]["value"] for l in target_labels}
    data.update({observation_label: y.unsqueeze(0)})

    # Condition on response and new thetas
    conditional_model = pyro.condition(model, data=data)
    trace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    trace.compute_log_prob()
    marginal_lp = logsumexp(trace.nodes[observation_label]["log_prob"], 0) - np.log(M)

    return (conditional_lp - marginal_lp).sum(0)/N


def donsker_varadhan_loss(model, observation_label, T):
    """
    Donsker-Varadhan estimate of the expected information gain (EIG).

    The Donsker-Varadhan representation of EIG is

        :math:`\\sup_T E[T(y, \\theta)] - \\log E[\\exp(T(\\bar{y}, \\bar{\\theta}))]`

    where the first expectation is over the joint :math:`p(y | \\theta, d)` and
    the second is over :math:`p(\\bar{y}|d)p(\\bar{\\theta})``.

    :param function model: A stochastic function.
    :param str observation_label: String label for observed variable.
    :param function or torch.nn.Module T: optimisable function `T` for use in the
        Donsker-Varadhan loss function.
    """

    ewma_log = EwmaLog(alpha=0.90)

    try:
        pyro.module("T", T)
    except AssertionError:
        pass

    def loss_fn(design, num_particles):

        expanded_design = design.expand((num_particles,) + design.shape)

        # Unshuffled data
        unshuffled_trace = poutine.trace(model).get_trace(expanded_design)
        y = unshuffled_trace.nodes[observation_label]["value"]

        # Shuffled data
        # Not actually shuffling, resimulate for safety
        data = {observation_label: y}
        conditional_model = pyro.condition(model, data=data)
        shuffled_trace = poutine.trace(conditional_model).get_trace(expanded_design)

        T_unshuffled = T(expanded_design, unshuffled_trace, observation_label)
        T_shuffled = T(expanded_design, shuffled_trace, observation_label)

        unshuffled_expectation = T_unshuffled.sum(0)/num_particles

        A = T_shuffled - np.log(num_particles)
        s, _ = torch.max(A, dim=0)
        shuffled_expectation = s + ewma_log((A - s).exp().sum(dim=0), s)

        loss = unshuffled_expectation - shuffled_expectation
        # Switch sign, sum over batch dimensions for scalar loss
        agg_loss = -loss.sum()
        return agg_loss, loss

    return loss_fn


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
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

    def __init__(self, alpha):
        self.alpha = alpha
        self.ewma = torch.tensor(0.)
        self.n = 0
        self.s = 0.

    def forward(self, inputs, s, dim=0, keepdim=False):
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
        return grad_output/self.ewma, None, None, None
