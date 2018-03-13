from __future__ import absolute_import, division, print_function

import torch
import warnings

import pyro
import pyro.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import MultiFrameTensor, get_iarange_stacks, TreeSum
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, is_nan


def _compute_upstream_grads(trace):
    upstream_grads = TreeSum()

    for site in trace.nodes.values():
        if site["type"] == "sample" and \
           not site["is_observed"] and \
           not getattr(site["fn"], "reparameterized", False):
            upstream_grads.add(site["cond_indep_stack"],
                               site["batch_log_pdf"] / site["scale"])

    return upstream_grads


def _compute_global_mft(trace):
    upstream_grads = MultiFrameTensor()

    for site in trace.nodes.values():
        if site["type"] == "sample" and \
           not site["is_observed"] and \
           not getattr(site["fn"], "reparameterized", False):
            upstream_grads.add((site["cond_indep_stack"],
                                site["batch_log_pdf"] / site["scale"]))

    return upstream_grads


def _compute_upstream_from_observe(model_trace, guide_trace, name):
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
    return set([stop_name])


def _compute_upstream_sample_sites(guide_trace, stop_names):
    for name, node in guide_trace.nodes.items():
        if node["type"] == "sample" and \
           not node["is_observed"] and \
           not getattr(node["fn"], "reparameterized", False):
            yield name, node
            stop_names.discard(name)
        if len(stop_names) == 0:
            break


def _magicbox_trace(model_trace, guide_trace, name, mft=None):
    """
    Computes the magicbox operator on an ELBO cost node
    """
    # first, if the cost node is observed,
    # find its most recent latent ancestor in guide_trace
    # stop_name = name
    if model_trace.nodes[name]["is_observed"]:
        stop_names = _compute_upstream_from_observe(model_trace, guide_trace, name)
    else:
        stop_names = set([name])

    # now compute the sum of log-probs of all upstream non-reparameterized nodes
    stacks = get_iarange_stacks(model_trace)
    if mft is None:
        log_prob_mft = MultiFrameTensor()  # MultiFrameTensor is awesome!
        for name2, node in _compute_upstream_sample_sites(guide_trace, stop_names):
            if node["type"] == "sample" and \
               not node["is_observed"] and \
               not getattr(node["fn"], "reparameterized", False):
                log_prob_mft.add((stacks[name2],
                                  # XXX hack to unscale gradient estimator
                                  node["batch_log_pdf"] / node["scale"]))
    else:
        log_prob_mft = mft

    # sum the multiframetensor to the cost node's indep stack
    s = log_prob_mft.sum_to(model_trace.nodes[name]["cond_indep_stack"])
    if s is None:
        # there were no non-reparameterized nodes, so just return 1
        return torch.ones(torch.Size())
    else:
        return torch.exp(s - s.detach())


def _magicbox_trace2(model_trace, guide_trace, name, mft=None):
    """
    Computes the magicbox operator on an ELBO cost node
    This version uses TreeSum instead of MultiFrameTensor
    """
    # now compute the of log-probs of all upstream non-reparameterized nodes
    if mft is None:
        log_prob_mft = _compute_upstream_grads(guide_trace)
    else:
        log_prob_mft = mft

    # sum to the cost node's indep stack
    s = log_prob_mft.get_upstream(model_trace.nodes[name]["cond_indep_stack"])
    if s is None:
        # there were no non-reparameterized nodes, so just return 1
        return torch.ones(torch.Size())
    else:
        return torch.exp(s - s.detach())


class Dice_ELBO(ELBO):

    def _get_traces(self, model, guide, *args, **kwargs):

        # identical to Trace_ELBO._get_traces
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
        Algorithm (naive):
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

            trace_mft = _compute_global_mft(guide_trace)
            # trace_mft2 = _compute_upstream_grads(guide_trace)

            for name, model_site in model_trace.nodes.items():
                if model_site["type"] == "sample":
                    mb_term = _magicbox_trace(model_trace, guide_trace, name,
                                              mft=trace_mft)
                    # mb_term = _magicbox_trace2(model_trace, guide_trace, name,
                    #                            mft=trace_mft)
                    site_cost = model_site["batch_log_pdf"]
                    if not model_site["is_observed"]:
                        site_cost = site_cost - guide_trace.nodes[name]["batch_log_pdf"]
                    elbo_particle = elbo_particle + torch.sum(mb_term * site_cost)
            # collect parameters to train from model and guide
            trainable_params = set(site["value"]
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values()
                                   if site["type"] == "param")

            elbo_particle = elbo_particle / self.num_particles
            elbo = elbo + elbo_particle.item()

            if trainable_params:
                (-elbo_particle).backward()
                pyro.get_param_store().mark_params_active(trainable_params)

        return -elbo

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = (model_trace.log_pdf() - guide_trace.log_pdf()).item()
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
