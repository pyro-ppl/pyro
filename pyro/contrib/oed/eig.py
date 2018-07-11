import torch
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI, Trace_ELBO


def guide_entropy(guide, *args):
    # TODO: strong assumptions being made here!
    trace = poutine.util.prune_subsample_sites(poutine.trace(guide).get_trace(*args))
    entropy = 0.
    for name, site in trace.nodes.items():
        if site["type"] == "sample":
            entropy += site["fn"].entropy()
    return entropy.squeeze() # .squeeze() necessary?


def vi_eig(model, design, observation_labels, vi_parameters, is_parameters):
    """Estimates the expected information gain (EIG) using variational inference (VI).

    The EIG is defined (up to linear rescaling) as

        :math:`EIG(d)=E_{Y\sim p(y|\theta, d)}[H(p(\theta|y, d))]`

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param observation_labels list or str: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a 
        posterior is to be inferred.
    :param dict vi_parameters: Variational inference parameters which should include:
        `optim`: an instance of `pyro.Optim`, `guide`: a guide function 
        compatible with `model`, `num_steps`: the number of VI steps to make
    :param dict is_parameters: Importance sampling parameters for the
        marginal distribution of :math:`Y`. May include `num_samples`: the number
        of samples to draw from the marginal.

    """

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]

    def posterior_entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        y_dict = {label: y[i, ...] for i, label in enumerate(observation_labels)}
        conditioned_model = pyro.condition(model, data=y_dict)
        posterior = SVI(conditioned_model, vi_parameters["guide"], 
            vi_parameters["optim"], loss=Trace_ELBO())
        with pyro.poutine.block():
            for _ in range(vi_parameters["num_steps"]):
                posterior.step(design)
        # Recover the entropy
        return guide_entropy(vi_parameters["guide"], design)

    y_dist = EmpiricalMarginal(
        Importance(model, num_samples=is_parameters.get("num_samples", None)).run(design), 
        sites=observation_labels)

    # Calculate the expected posterior entropy under this distn of y
    loss_dist = EmpiricalMarginal(Search(posterior_entropy).run(y_dist, design))
    loss = loss_dist.mean

    return loss

