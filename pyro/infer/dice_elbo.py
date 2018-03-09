from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import MultiFrameTensor
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match


def _magicbox_trace(model_trace, guide_trace, name):
    """
    TODO docs
    """
    stop_name = name
    stacks = model_trace.graph["iarange_info"]["iarange_stacks"]
    if model_trace.nodes[name]["is_observed"]:
        front = set()
        for name2, node in model_trace.nodes.items():
            if node["type"] == "sample" and \
               not node["is_observed"]:
                front.add(name2)
                if name2 == name:
                    break
        for name2 in guide_trace.nodes:
            if name2 in front:
                front.discard(name2)
            if len(front) == 0:
                stop_name = name2
                break

    log_prob_mft = MultiFrameTensor()
    for name2, node in guide_trace.nodes.items():
        if node["type"] == "sample" and \
           not node["is_observed"] and \
           not getattr(node["fn"], "reparameterized", False):
            log_prob_mft.add((stacks[name2], node["batch_log_pdf"]))
        if name2 == stop_name:
            break

    s = log_prob_mft.sum_to(model_trace.nodes[name]["cond_indep_stack"])
    if s is None:
        return torch.ones(1)
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
            guide_trace.compute_score_parts()

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
                    model_term = model_site["batch_log_pdf"]
                    if model_site["is_observed"]:
                        mb_term = _magicbox_trace(model_trace, guide_trace, name)
                        elbo_particle = elbo_particle + torch.sum(mb_term * model_term)
                    else:
                        mb_term = _magicbox_trace(model_trace, guide_trace, name)
                        guide_term = guide_trace.nodes[name]["batch_log_pdf"] * -1
                        elbo_particle = elbo_particle + torch.sum(mb_term * model_term)
                        elbo_particle = elbo_particle + torch.sum(mb_term * guide_term)
            # collect parameters to train from model and guide
            trainable_params = set(site["value"]
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values()
                                   if site["type"] == "param")

            elbo = elbo + elbo_particle.detach() / self.num_particles

            if trainable_params:
                (elbo_particle / self.num_particles).backward()
                pyro.get_param_store().mark_params_active(trainable_params)

        return -elbo

    def loss(self, model, guide, *args, **kwargs):
        _loss = self.loss(model, guide, *args, **kwargs)
        return _loss
