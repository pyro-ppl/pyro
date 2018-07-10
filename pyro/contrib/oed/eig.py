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

