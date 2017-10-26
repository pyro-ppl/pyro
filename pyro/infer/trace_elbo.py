import numbers
import warnings

import pyro
import pyro.poutine as poutine
from pyro.infer.enum import iter_discrete_traces
from pyro.distributions.util import torch_zeros_like


class Trace_ELBO(object):
    """
    A trace implementation of ELBO-based SVI
    """
    def __init__(self,
                 num_particles=1,
                 enum_discrete=False):
        """
        :param num_particles: the number of particles/samples used to form the ELBO (gradient) estimators
        :param bool enum_discrete: whether to sum over discrete latent variables, rather than sample them
        """
        super(Trace_ELBO, self).__init__()
        self.num_particles = num_particles
        self.enum_discrete = enum_discrete

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator

        XXX support for automatically settings args/kwargs to volatile?
        """

        for i in range(self.num_particles):
            if self.enum_discrete:
                # This iterates over a bag of traces, for each particle.
                for scale, guide_trace in iter_discrete_traces("flat", guide, *args, **kwargs):
                    model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                                graph_type="flat").get_trace(*args, **kwargs)
                    log_r = model_trace.batch_log_pdf() - guide_trace.batch_log_pdf()
                    weight = scale / self.num_particles
                    yield weight, model_trace, guide_trace, log_r
                continue

            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace(*args, **kwargs)
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
        samples_seen = set()
        for weight, model_trace, guide_trace, log_r in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = weight * 0

            log_pdf = "batch_log_pdf" if self.enum_discrete else "log_pdf"
            for name in model_trace.nodes.keys():
                if model_trace.nodes[name]["type"] == "sample":
                    if model_trace.nodes[name]["is_observed"]:
                        elbo_particle += model_trace.nodes[name][log_pdf]
                    else:
                        samples_seen.add(name)
                        elbo_particle += model_trace.nodes[name][log_pdf]
                        elbo_particle -= guide_trace.nodes[name][log_pdf]

            # drop terms of weight zero to avoid nans
            if isinstance(weight, numbers.Number):
                if weight == 0.0:
                    elbo_particle = torch_zeros_like(elbo_particle)
            else:
                elbo_particle[weight == 0] = 0.0

            elbo += (weight * elbo_particle).data.sum()

        all_samples = [name for name in guide_trace.nodes.keys() \
                       if guide_trace.nodes[name]["type"] == "sample"]

        if samples_seen != set(all_samples):
            extra_vars = set(all_samples).difference(samples_seen)
            warnings.warn('Model and guide samples do not match up: {}'.format(extra_vars))

        loss = -elbo
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        elbo = 0.0
        surrogate_elbo = 0.0
        trainable_params = set()
        samples_seen = set()
        # grab a trace from the generator
        for weight, model_trace, guide_trace, log_r in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = weight * 0
            surrogate_elbo_particle = weight * 0
            # compute elbo and surrogate elbo
            log_pdf = "batch_log_pdf" if self.enum_discrete else "log_pdf"
            for name in model_trace.nodes.keys():
                if model_trace.nodes[name]["type"] == "sample":
                    if model_trace.nodes[name]["is_observed"]:
                        elbo_particle += model_trace.nodes[name][log_pdf]
                        surrogate_elbo_particle += model_trace.nodes[name][log_pdf]
                    else:
                        samples_seen.add(name)
                        lp_lq = model_trace.nodes[name][log_pdf] - guide_trace.nodes[name][log_pdf]
                        elbo_particle += lp_lq
                        if guide_trace.nodes[name]["fn"].reparameterized:
                            surrogate_elbo_particle += lp_lq
                        else:
                            # XXX should the user be able to control inclusion of the -logq term below?
                            surrogate_elbo_particle += model_trace.nodes[name][log_pdf] + \
                                log_r.detach() * guide_trace.nodes[name][log_pdf]

            # drop terms of weight zero to avoid nans
            if isinstance(weight, numbers.Number):
                if weight == 0.0:
                    elbo_particle = torch_zeros_like(elbo_particle)
                    surrogate_elbo_particle = torch_zeros_like(surrogate_elbo_particle)
            else:
                weight_eq_zero = (weight == 0)
                elbo_particle[weight_eq_zero] = 0.0
                surrogate_elbo_particle[weight_eq_zero] = 0.0

            elbo += (weight * elbo_particle).data.sum()
            surrogate_elbo += (weight * surrogate_elbo_particle).sum()

            # grab model parameters to train
            for name in model_trace.nodes.keys():
                if model_trace.nodes[name]["type"] == "param":
                    trainable_params.add(model_trace.nodes[name]["value"])

            # grab guide parameters to train
            for name in guide_trace.nodes.keys():
                if guide_trace.nodes[name]["type"] == "param":
                    trainable_params.add(guide_trace.nodes[name]["value"])

        all_samples = [name for name in guide_trace.nodes.keys() \
                       if guide_trace.nodes[name]["type"] == "sample"]

        if samples_seen != set(all_samples):
            extra_vars = set(all_samples).difference(samples_seen)
            warnings.warn('Model and guide samples do not match up: {}'.format(extra_vars))

        surrogate_loss = -surrogate_elbo
        surrogate_loss.backward()
        loss = -elbo

        pyro.get_param_store().mark_params_active(trainable_params)

        return loss
