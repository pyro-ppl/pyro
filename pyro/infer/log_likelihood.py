from typing import Callable, Dict

import torch

import pyro.poutine
from pyro.infer import Predictive


def log_likelihood(
    model: Callable,
    posterior_samples: Dict[str, torch.Tensor] = None,
    guide: Callable = None,
    num_samples: int = None,
    parallel: bool = True,
) -> Callable:
    """Calculates the log likelihood of the data given posterior samples (or a
    guide that can generate them).

    :param model: Model function containing pyro primitives. Must have observed
        sites conditioned on data.
    :type model: Callable
    :param posterior_samples: Dictionary of posterior samples, defaults to None
    :type posterior_samples: Dict[str, torch.Tensor], optional
    :param guide: Guide function, defaults to None
    :type guide: Callable, optional
    :param num_samples: Number of posterior samples to evaluate, defaults to None
    :type num_samples: int, optional
    :param parallel: Whether to use vectorization, defaults to True
    :type parallel: bool, optional
    :param pointwise: Whether to return pointwise log likelihood (2D tensor),
        or total (1D tensor); in both cases dim=0 is sample dim, defaults to True
    :rtype: Callable
    """

    # define the function to return the likelihood values
    def log_like_helper(*args, **kwargs) -> torch.Tensor:
        # trace once to get observed sites
        trace = pyro.poutine.trace(model).get_trace(*args, **kwargs)
        observed = {
            name: site["value"]
            for name, site in trace.nodes.items()
            if site["type"] == "sample" and site["is_observed"]
        }
        # now trace it and extract the likelihood from observed sites
        predictive = Predictive(
            model, posterior_samples, guide, num_samples, (), parallel)
        # use vectorized trace
        if parallel:
            log_like = dict()
            trace = predictive.get_vectorized_trace(*args, **kwargs)
            for obs_name, obs_val in observed.items():
                obs_site = trace.nodes[obs_name]
                log_like[obs_name] = obs_site["fn"].log_prob(obs_val).detach().cpu()
        # iterate over samples from posterior if model can't be vectorized
        else:
            all_samples = {
                k: v for k, v in predictive(*args, **kwargs).items()
                if k not in observed}
            log_like = dict()
            for i in range(num_samples):
                param_sample = {k: v[i] for k, v in all_samples.items()}
                cond_model = pyro.condition(model, data=param_sample)
                traced_model = pyro.poutine.trace(cond_model)
                trace = traced_model.get_trace(*args, **kwargs)
                for obs_name, obs_val in observed.items():
                    obs_site = trace.nodes[obs_name]
                    ll_sample = obs_site["fn"].log_prob(obs_val).detach().cpu()
                    log_like.get(obs_name, []).append(ll_sample)
            log_like = {
                obs_name: torch.stack(ll_list, dim=0)
                for obs_name, ll_list in log_like.items()
            }
        return log_like
    return log_like_helper