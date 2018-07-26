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


def naiveRainforth(model, design, *args, observation_labels="y", N=100, M=10):

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]

    # 100 traces using batching
    eig = 0.
    for _ in range(N):
        y_given_theta = 0.
        y = {}
        trace = poutine.trace(model).get_trace(design)
        trace.compute_log_prob()
        for label in observation_labels:
            # Valid? Yes, this is log probability conditional on
            # theta, and any previously sampled y's
            # Order doesn't matter
            y_given_theta += trace.nodes[label]["log_prob"]
            y[label] = trace.nodes[label]["value"]

        lp_shape = y_given_theta.shape

        y_given_other_theta = torch.zeros(*lp_shape, M+1)
        y_given_other_theta[..., -1] = y_given_theta
        conditional_model = pyro.condition(model, data=y)
        for j in range(M):
            trace = poutine.trace(conditional_model).get_trace(design)
            trace.compute_log_prob()
            for label in observation_labels:
                y_given_other_theta[..., j] += trace.nodes[label]["log_prob"]

        eig += y_given_theta - torch.distributions.utils.log_sum_exp(
            y_given_other_theta).squeeze() + np.log(M)

    return eig/N


def donsker_varadhan_loss(model, design, observation_label, target_label,
                          num_particles, T):

    global i, y_samples, theta_samples, theta_shuffled_samples, ewma, alpha

    loss = 0.
    trace = poutine.trace(model).get_trace(design)
    trace.compute_log_prob()
    y = trace.nodes[observation_label]["value"]
    theta = trace.nodes[target_label]["value"]

    y_samples = y.new_empty((num_particles, *y.shape))
    theta_samples = theta.new_empty((num_particles, *theta.shape))

    y_samples[0, ...] = y
    theta_samples[0, ...] = theta
    
    for i in range(1, num_particles):
        trace = poutine.trace(model).get_trace(design)
        y = trace.nodes[observation_label]["value"]
        theta = trace.nodes[target_label]["value"]
        y_samples[i, ...] = y
        theta_samples[i, ...] = theta

    idx = torch.randperm(num_particles)
    theta_shuffled_samples = theta_samples[idx, ...]

    pyro.module("T", T)

    i= 0
    ewma = None
    alpha = 10.

    def loss_fn():

        global i, y_samples, theta_samples, theta_shuffled_samples, ewma, alpha

        fvals = T(y_samples, theta_samples, design)
        fshuffled = T(y_samples, theta_shuffled_samples, design)

        expect_exp = logsumexp(fshuffled, dim=0) - np.log(num_particles)
        if ewma is None:
            ewma = torch.exp(expect_exp)
        else:
            ewma = (1/(1+alpha))*(torch.exp(expect_exp) + alpha*ewma)
        expect_exp.grad = 1./ewma
        loss = torch.sum(fvals, 0)/num_particles - expect_exp
        # loss = torch.sum(fvals, 0)/num_particles - \
        #     torch.sum(torch.exp(fshuffled-1.), 0)/num_particles

        for _ in range(20):
            trace = poutine.trace(model).get_trace(design)
            y = trace.nodes[observation_label]["value"]
            theta = trace.nodes[target_label]["value"]
            y_samples[i, ...] = y
            theta_samples[i, ...] = theta
            i = (i+1)%num_particles
        idx = torch.randperm(num_particles)
        theta_shuffled_samples = theta_samples[idx, ...]

        # Switch sign, sum over batch dimensions for scalar loss
        print(loss)
        agg_loss = -loss.sum()
            
        return agg_loss

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
