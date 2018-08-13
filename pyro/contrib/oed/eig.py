import torch
import numpy as np

import pyro
from pyro import poutine
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI
from pyro.contrib.autoguide import mean_field_guide_entropy


def vi_ape(model, design, observation_labels, vi_parameters, is_parameters):
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
    :return: Loss function estimate
    :rtype: `torch.Tensor`

    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]

    def posterior_entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        y_dict = {label: y[i, ...] for i, label in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        SVI(conditioned_model, **vi_parameters).run(design)
        # Recover the entropy
        return mean_field_guide_entropy(vi_parameters["guide"], design)

    y_dist = EmpiricalMarginal(Importance(model, **is_parameters).run(design),
                               sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    loss = loss_dist.mean

    return loss


def naive_rainforth(model, design, observation_label="y", target_label="theta",
                    N=100, M=10):

    expanded_design = design.expand((N, *design.shape))
    trace = poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    y = trace.nodes[observation_label]["value"]
    conditional_lp = trace.nodes[observation_label]["log_prob"]

    reexpanded_design = design.expand((M, 1, *design.shape))
    reexp_trace = poutine.trace(model).get_trace(reexpanded_design)
    marginal_lp = cond_log_prob(model, y, reexp_trace.nodes[target_label]["value"],
                                observation_label, target_label, design)[0]

    return (conditional_lp - marginal_lp).sum(0)/N


def donsker_varadhan_loss(model, observation_label, target_label, U):

    ewma_log = EwmaLog(alpha=0.66)

    pyro.module("U", U)

    def loss_fn(design, num_particles):

        expanded_design = design.expand((num_particles, *design.shape))

        trace = poutine.trace(model).get_trace(expanded_design)
        y = trace.nodes[observation_label]["value"]
        theta = trace.nodes[target_label]["value"]

        # Compute log probabilities
        trace.compute_log_prob()
        unshuffled_lp = trace.nodes[observation_label]["log_prob"]
        # Not actually shuffling, resimulate for safety
        shuffled_lp, _ = cond_log_prob(model, y, None, observation_label, 
                                       target_label, expanded_design.unsqueeze(0))

        T_unshuffled = U(expanded_design, y, unshuffled_lp)
        T_shuffled = U(expanded_design, y, shuffled_lp)

        A = T_shuffled - np.log(num_particles)
        s, _ = torch.max(A, dim=0)
        expect_exp = s + ewma_log((A - s).exp().sum(dim=0), s)
        # expect_exp = logsumexp(A, dim=0)

        # Switch sign, sum over batch dimensions for scalar loss
        loss = T_unshuffled.sum(0)/num_particles - expect_exp
        agg_loss = -loss.sum()
        return agg_loss, loss

    return loss_fn


def cond_log_prob(model, observation, target, observation_label, target_label, *args):
    if target is not None:
        M = target.shape[0]
        conditional_model = pyro.condition(model, data={
            observation_label: observation.unsqueeze(0),
            target_label: target
            })
    else:
        M = 1
        conditional_model = pyro.condition(model, data={
            observation_label: observation.unsqueeze(0),
            })
    trace = poutine.trace(conditional_model).get_trace(*args)
    trace.compute_log_prob()
    return (logsumexp(trace.nodes[observation_label]["log_prob"], 0) - np.log(M), 
            trace.nodes[target_label]["value"])


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
