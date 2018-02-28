from __future__ import absolute_import, division, print_function

import math
import numbers
import warnings

import torch

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.util import MultiViewTensor
from pyro.poutine.enumerate_poutine import EnumeratePoutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, is_nan


def _compute_log_r(model_trace, guide_trace):
    """
    Constructs a MultiViewTensor representing detached total log(p) - log(q).
    """
    log_r = MultiViewTensor()
    for site in model_trace.nodes.values():
        if site["type"] == "sample":
            log_r.add(site["batch_log_pdf"].detach())
    for site in guide_trace.nodes.values():
        if site["type"] == "sample":
            log_r.add(-site["batch_log_pdf"].detach())
    return log_r


def _idot(weight, term, iarange_dims):
    """
    Computes weighted sum of a tensor, accounting for enumeration dims on the
    left and iarange dims on the right.
    """
    if isinstance(weight, numbers.Number):
        if weight == 0:
            return term.new([0]).squeeze()
        return weight * term.sum()

    if weight.dim() > term.dim():
        term = term.expand(torch.Size((1,) * (weight.dim() - term.dim())) + term.shape)
    elif weight.dim() < term.dim():
        weight = weight.expand(torch.Size((1,) * (term.dim() - weight.dim())) + weight.shape)
    assert weight.dim() == term.dim()

    enumerate_dims = weight.dim() - iarange_dims
    for i, (weight_size, term_size) in enumerate(zip(weight.shape, term.shape)):
        if weight_size > term_size:
            if i < enumerate_dims:
                # sum-out irrelevant enumeration dimensions
                weight = weight.sum(i, True)
            else:
                # arbitrarily select-out irrelevant iarange dimensions
                weight = weight[(slice(None),) * i + (0,)]

    prod = weight * term
    prod[(weight == 0).expand_as(prod)] = 0
    return prod.sum()


class Trace_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI
    """

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """
        # enable parallel enumeration
        guide = EnumeratePoutine(guide, first_available_dim=self.max_iarange_nesting)

        for i in range(self.num_particles):
            for scale, guide_trace in iter_discrete_traces("flat", self.max_iarange_nesting, guide, *args, **kwargs):
                model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                            graph_type="flat").get_trace(*args, **kwargs)

                check_model_guide_match(model_trace, guide_trace)
                guide_trace = prune_subsample_sites(guide_trace)
                model_trace = prune_subsample_sites(model_trace)

                model_trace.compute_batch_log_pdf()
                guide_trace.compute_score_parts()
                weight = scale / self.num_particles
                yield weight, model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0

        for weight, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    elbo += _idot(weight, site["batch_log_pdf"], self.max_iarange_nesting).detach().item()
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    elbo -= _idot(weight, site["batch_log_pdf"], self.max_iarange_nesting).detach().item()

        loss = -elbo
        if math.isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        elbo = 0.0
        # grab a trace from the generator
        for weight, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0
            log_r = None  # compute lazily

            # compute elbo and surrogate elbo
            for name, model_site in model_trace.nodes.items():
                if model_site["type"] == "sample":
                    model_log_pdf = _idot(weight, model_site["batch_log_pdf"], self.max_iarange_nesting)
                    if model_site["is_observed"]:
                        elbo_particle += model_log_pdf.detach()
                        surrogate_elbo_particle += model_log_pdf
                    else:
                        guide_site = guide_trace.nodes[name]
                        guide_log_pdf, score_function_term, entropy_term = guide_site["score_parts"]

                        guide_log_pdf = _idot(weight, guide_log_pdf, self.max_iarange_nesting)
                        elbo_particle += model_log_pdf.detach() - guide_log_pdf.detach()
                        surrogate_elbo_particle += model_log_pdf

                        if not is_identically_zero(entropy_term):
                            surrogate_elbo_particle -= _idot(weight, entropy_term, self.max_iarange_nesting)

                        if not is_identically_zero(score_function_term):
                            if log_r is None:
                                log_r = _compute_log_r(model_trace, guide_trace)
                            score_function = log_r.contract_to(score_function_term) * score_function_term
                            surrogate_elbo_particle += _idot(weight, score_function, self.max_iarange_nesting)

            if not is_identically_zero(elbo_particle):
                elbo += elbo_particle.item()

            if not is_identically_zero(surrogate_elbo_particle):
                trainable_params = set(site["value"]
                                       for trace in (model_trace, guide_trace)
                                       for site in trace.nodes.values()
                                       if site["type"] == "param")
                if trainable_params:
                    surrogate_loss_particle = -surrogate_elbo_particle
                    surrogate_loss_particle.backward()
                    pyro.get_param_store().mark_params_active(trainable_params)

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
