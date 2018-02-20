from __future__ import absolute_import, division, print_function

import numbers
import warnings

import torch

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero, sum_rightmost
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.util import torch_backward, torch_data_sum, torch_sum
from pyro.poutine.enumerate_poutine import EnumeratePoutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, is_nan


class Trace_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI
    """

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """

        for i in range(self.num_particles):
            if self.enum_discrete:
                # This enables parallel enumeration.
                guide = EnumeratePoutine(guide, first_available_dim=self.max_iarange_nesting)
                # This iterates over a bag of traces, one trace per particle.
                for scale, guide_trace in iter_discrete_traces("flat", self.max_iarange_nesting,
                                                               guide, *args, **kwargs):
                    model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                                graph_type="flat").get_trace(*args, **kwargs)

                    check_model_guide_match(model_trace, guide_trace)
                    guide_trace = prune_subsample_sites(guide_trace)
                    model_trace = prune_subsample_sites(model_trace)

                    log_r = 0
                    model_trace.compute_batch_log_pdf()
                    for site in model_trace.nodes.values():
                        if site["type"] == "sample":
                            log_r += sum_rightmost(site["batch_log_pdf"], self.max_iarange_nesting)
                    guide_trace.compute_score_parts()
                    for site in guide_trace.nodes.values():
                        if site["type"] == "sample":
                            log_r -= sum_rightmost(site["batch_log_pdf"], self.max_iarange_nesting)

                    weight = scale / self.num_particles
                    yield weight, model_trace, guide_trace, log_r
                continue

            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace(*args, **kwargs)

            check_model_guide_match(model_trace, guide_trace)
            guide_trace = prune_subsample_sites(guide_trace)
            model_trace = prune_subsample_sites(model_trace)

            guide_trace.compute_score_parts()
            log_r = model_trace.log_pdf() - guide_trace.log_pdf()
            weight = 1.0 / self.num_particles
            yield weight, model_trace, guide_trace, log_r

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for weight, model_trace, guide_trace, log_r in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = weight * 0
            for name, model_site in model_trace.nodes.items():
                if model_site["type"] == "sample":
                    model_log_pdf = sum_rightmost(model_site["batch_log_pdf"], self.max_iarange_nesting)
                    if model_site["is_observed"]:
                        elbo_particle += model_log_pdf
                    else:
                        guide_site = guide_trace.nodes[name]
                        guide_log_pdf = sum_rightmost(guide_site["batch_log_pdf"], self.max_iarange_nesting)
                        elbo_particle += model_log_pdf - guide_log_pdf

            # drop terms of weight zero to avoid nans
            if isinstance(weight, numbers.Number):
                if weight == 0.0:
                    elbo_particle = torch.zeros_like(elbo_particle)
            else:
                elbo_particle[weight == 0] = 0.0

            elbo += torch_data_sum(weight * elbo_particle)

        loss = -elbo
        if is_nan(loss):
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
        for weight, model_trace, guide_trace, log_r in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = weight * 0
            surrogate_elbo_particle = weight * 0
            # compute elbo and surrogate elbo
            for name, model_site in model_trace.nodes.items():
                if model_site["type"] == "sample":
                    model_log_pdf = sum_rightmost(model_site["batch_log_pdf"], self.max_iarange_nesting)
                    if model_site["is_observed"]:
                        elbo_particle += model_log_pdf
                        surrogate_elbo_particle += model_log_pdf
                    else:
                        guide_site = guide_trace.nodes[name]
                        guide_log_pdf, score_function_term, entropy_term = guide_site["score_parts"]

                        guide_log_pdf = sum_rightmost(guide_log_pdf, self.max_iarange_nesting)
                        elbo_particle += model_log_pdf - guide_log_pdf
                        surrogate_elbo_particle += model_log_pdf

                        if not is_identically_zero(entropy_term):
                            entropy_term = sum_rightmost(entropy_term, self.max_iarange_nesting)
                            surrogate_elbo_particle -= entropy_term

                        if not is_identically_zero(score_function_term):
                            score_function_term = sum_rightmost(score_function_term, self.max_iarange_nesting)
                            surrogate_elbo_particle += log_r.detach() * score_function_term

            # drop terms of weight zero to avoid nans
            if isinstance(weight, numbers.Number):
                if weight == 0.0:
                    elbo_particle = torch.zeros_like(elbo_particle)
                    surrogate_elbo_particle = torch.zeros_like(surrogate_elbo_particle)
            else:
                weight_eq_zero = (weight == 0)
                elbo_particle[weight_eq_zero] = 0.0
                surrogate_elbo_particle[weight_eq_zero] = 0.0

            elbo += torch_data_sum(weight * elbo_particle)
            surrogate_elbo_particle = torch_sum(weight * surrogate_elbo_particle)

            # collect parameters to train from model and guide
            trainable_params = set(site["value"]
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values()
                                   if site["type"] == "param")

            if trainable_params:
                surrogate_loss_particle = -surrogate_elbo_particle
                torch_backward(surrogate_loss_particle)
                pyro.get_param_store().mark_params_active(trainable_params)

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
