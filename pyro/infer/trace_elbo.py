from __future__ import absolute_import, division, print_function

import math
import warnings

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.util import MultiViewTensor
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
        # enable parallel enumeration
        guide = EnumeratePoutine(guide, first_available_dim=self.max_iarange_nesting)

        for i in range(self.num_particles):
            # This iterates over a bag of traces, for each particle.
            for guide_trace in iter_discrete_traces("flat", self.max_iarange_nesting, guide, *args, **kwargs):
                model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                            graph_type="flat").get_trace(*args, **kwargs)

                check_model_guide_match(model_trace, guide_trace)
                guide_trace = prune_subsample_sites(guide_trace)
                model_trace = prune_subsample_sites(model_trace)

                model_trace.compute_batch_log_pdf()
                guide_trace.compute_score_parts()

                log_r = MultiViewTensor()
                for site in model_trace.nodes.values():
                    if site["type"] == "sample":
                        log_r.add(site["batch_log_pdf"].detach())
                for site in guide_trace.nodes.values():
                    if site["type"] == "sample":
                        log_r.add(-site["batch_log_pdf"].detach())

                yield model_trace, guide_trace, log_r

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
                    elbo += site["batch_log_pdf"].detach()
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    elbo -= site["batch_log_pdf"].detach()

        loss = -(weight * elbo).sum().item()
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
        for model_trace, guide_trace, log_r in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0
            # compute elbo and surrogate elbo
            for name, model_site in model_trace.nodes.items():
                if model_site["type"] == "sample":
                    if model_site["is_observed"]:
                        elbo_particle += model["log_pdf"].item()
                        surrogate_elbo_particle = surrogate_elbo_particle + model["log_pdf"]
                    else:
                        guide_site = guide_trace.nodes[name]
                        guide_log_pdf, score_function_term, entropy_term = guide_site["score_parts"]

                        elbo_particle += model["log_pdf"].item() - guide["log_pdf"].item()
                        surrogate_elbo_particle = surrogate_elbo_particle + model["log_pdf"]

                        if not is_identically_zero(entropy_term):
                            surrogate_elbo_particle = surrogate_elbo_particle - entropy_term.sum()

                        if not is_identically_zero(score_function_term):
                            score_function = log_r.contract_to(score_function_term) * score_function_term
                            surrogate_elbo_particle = surrogate_elbo_particle + score_function.sum()

            elbo += elbo_particle / self.num_particles

            if not is_identically_zero(surrogate_elbo_particle):
                trainable_params = set(site["value"]
                                       for trace in (model_trace, guide_trace)
                                       for site in trace.nodes.values()
                                       if site["type"] == "param")
                if trainable_params:
                    surrogate_loss_particle = -surrogate_elbo_particle / self.num_particles
                    surrogate_loss_particle.backward()
                    pyro.get_param_store().mark_params_active(trainable_params)

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
