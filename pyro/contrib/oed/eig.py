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


def naive_rainforth(model, design, *args, observation_label="y", target_label="theta",
                    N=100, M=10):

    expanded_design = design.expand((N, *design.shape))
    # Run model `num_particles` times vectorized
    trace = poutine.trace(model).get_trace(expanded_design)
    reexpanded_design = expanded_design.expand((M, *expanded_design.shape))
    reexpanded_y = trace.nodes[observation_label]["value"].expand(
        (M, *trace.nodes[observation_label]["value"].shape))
    conditional_model = pyro.condition(model, data={observation_label: reexpanded_y})
    c_trace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    c_trace.compute_log_prob()
    base_lp = logsumexp(c_trace.nodes[observation_label]["log_prob"], 0) - np.log(M)

    return (trace.nodes[observation_label]["log_prob"] - base_lp).sum(0)/N


def donsker_varadhan_loss(model, design, observation_label, target_label,
                          num_particles, U):

    global ewma
    ewma = None
    alpha = 10.

    expanded_design = design.expand((num_particles, *design.shape))
    # Run model `num_particles` times vectorized

    pyro.module("U", U)

    def loss_fn():

        global ewma

        re_n = 1

        trace = poutine.trace(model).get_trace(expanded_design)
        reexpanded_design = design.expand((re_n, *expanded_design.shape))
        reexpanded_y = trace.nodes[observation_label]["value"].expand(
            (re_n, *trace.nodes[observation_label]["value"].shape))
        conditional_model = pyro.condition(model, data={observation_label: reexpanded_y})
        c_trace = poutine.trace(conditional_model).get_trace(reexpanded_design)
        c_trace.compute_log_prob()
        base_lp = logsumexp(c_trace.nodes[observation_label]["log_prob"], 0)- np.log(re_n)
        rand_perm = torch.randperm(num_particles)
        shuffled_trace = shuffle_trace(model, trace, rand_perm, target_label, 
                                       observation_label, expanded_design)

        # Now compute the log_probs- avoid using replay
        trace.compute_log_prob()
        shuffled_trace.compute_log_prob()

        T_unshuffled = U(expanded_design, trace.nodes[observation_label]["value"],
                         trace.nodes[observation_label]["log_prob"])
        T_shuffled = U(expanded_design, shuffled_trace.nodes[observation_label]["value"],
                       shuffled_trace.nodes[observation_label]["log_prob"])

        # Use ewma correction to gradients
        expect_exp = logsumexp(T_shuffled, dim=0) - np.log(num_particles)
        if ewma is None:
            ewma = torch.exp(expect_exp)
        else:
            ewma = (1/(1+alpha))*(torch.exp(expect_exp) + alpha*ewma)
        expect_exp.grad = 1./ewma
        print('ewma', ewma)

        # Switch sign, sum over batch dimensions for scalar loss
        loss = torch.sum(T_unshuffled, 0)/num_particles - expect_exp
        # print(loss)
        agg_loss = -loss.sum()
        return agg_loss, loss

    return loss_fn


def shuffle_trace(model, trace, perm, target_label, observation_label, *args):
    conditional_model = pyro.condition(model, data={
        observation_label: trace.nodes[observation_label]["value"]
        })
    c_trace = poutine.trace(conditional_model).get_trace(*args)
    return c_trace


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
