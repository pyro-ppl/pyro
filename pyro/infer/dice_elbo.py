from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import MultiFrameTensor, get_iarange_stacks
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape


def _magicbox_trace(model_trace, guide_trace, name):
    """
    TODO docs
    """
    stop_name = name
    # stacks = get_iarange_stacks(model_trace)
    if model_trace.nodes[name]["is_observed"]:
        front = set()
        for name2, node in model_trace.nodes.items():
            if node["type"] == "sample" and \
               not node["is_observed"]:
                front.add(name2)
                if name2 == name:
                    break
        for name2 in guide_trace.nodes:
            front.discard(name2)
            if len(front) == 0:
                stop_name = name2
                break

    log_prob_mft = MultiFrameTensor()
    for name2, node in guide_trace.nodes.items():
        if node["type"] == "sample" and \
           not node["is_observed"] and \
           not getattr(node["fn"], "reparameterized", False):
            log_prob_mft.add((node["cond_indep_stack"], node["batch_log_pdf"] / node["scale"]))
        if name2 == stop_name:
            break

    s = log_prob_mft.sum_to(model_trace.nodes[name]["cond_indep_stack"])
    if s is None:
        return torch.ones(torch.Size())
    else:
        return torch.exp(s - s.detach())


class Dice_ELBO(ELBO):

    def _get_traces(self, model, guide, *args, **kwargs):
        for i in range(self.num_particles):
            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace(*args, **kwargs)

            check_model_guide_match(model_trace, guide_trace)

            guide_trace = prune_subsample_sites(guide_trace)
            model_trace = prune_subsample_sites(model_trace)

            model_trace.compute_batch_log_pdf()
            guide_trace.compute_batch_log_pdf()

            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)

            yield model_trace, guide_trace

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        Algorithm:
        Naive:
        For each cost node c:
          identify upstream nodes
          collect all independence contexts
          compute multiframetensor of log probs
          compute cost node c with appropriate shape from its context
          magicbox and sum
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0

            for name, model_site in model_trace.nodes.items():
                if model_site["type"] == "sample":
                    mb_term = _magicbox_trace(model_trace, guide_trace, name)
                    site_cost = model_site["batch_log_pdf"]
                    if not model_site["is_observed"]:
                        site_cost = site_cost - guide_trace.nodes[name]["batch_log_pdf"]
                    elbo_particle = elbo_particle + torch.sum(mb_term * site_cost)
            # collect parameters to train from model and guide
            trainable_params = set(site["value"]
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values()
                                   if site["type"] == "param")

            elbo = elbo + elbo_particle.detach() / self.num_particles

            if trainable_params:
                (-elbo_particle / self.num_particles).backward()
                pyro.get_param_store().mark_params_active(trainable_params)

        return -elbo

    def loss(self, model, guide, *args, **kwargs):
        _loss = self.loss(model, guide, *args, **kwargs)
        return _loss
