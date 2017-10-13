import pyro
import pyro.poutine as poutine


class Trace_ELBO(object):
    """
    A trace implementation of ELBO-based SVI
    """
    def __init__(self,
                 num_particles=1):
        """
        :param num_particles: the number of particles/samples used to form the ELBO (gradient) estimators
        """
        super(Trace_ELBO, self).__init__()
        self.num_particles = num_particles

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator

        XXX support for automatically settings args/kwargs to volatile?
        """

        for i in range(self.num_particles):
            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace(*args, **kwargs)
            log_r = model_trace.log_pdf() - guide_trace.log_pdf()
            yield model_trace, guide_trace, log_r

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace, log_r in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0.0

            for name in model_trace.nodes.keys():
                if model_trace.nodes[name]["type"] == "observe":
                    elbo_particle += model_trace.nodes[name]["log_pdf"]
                elif model_trace.nodes[name]["type"] == "sample":
                    elbo_particle += model_trace.nodes[name]["log_pdf"]
                    elbo_particle -= guide_trace.nodes[name]["log_pdf"]

            elbo += elbo_particle.data[0] / self.num_particles

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

        # grab a trace from the generator
        for model_trace, guide_trace, log_r in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0.0
            surrogate_elbo_particle = 0.0

            # compute elbo and surrogate elbo
            for name in model_trace.nodes.keys():
                if model_trace.nodes[name]["type"] == "observe":
                    elbo_particle += model_trace.nodes[name]["log_pdf"]
                    surrogate_elbo_particle += model_trace.nodes[name]["log_pdf"]
                elif model_trace.nodes[name]["type"] == "sample":
                    lp_lq = model_trace.nodes[name]["log_pdf"] - guide_trace.nodes[name]["log_pdf"]
                    elbo_particle += lp_lq
                    if model_trace.nodes[name]["fn"].reparameterized:
                        surrogate_elbo_particle += lp_lq
                    else:
                        # XXX should the user be able to control inclusion of the -logq term below?
                        surrogate_elbo_particle += model_trace.nodes[name]["log_pdf"] + \
                            log_r.detach() * guide_trace.nodes[name]["log_pdf"]

            elbo += elbo_particle.data[0] / self.num_particles
            surrogate_elbo += surrogate_elbo_particle / self.num_particles

            # grab model parameters to train
            for name in model_trace.nodes.keys():
                if model_trace.nodes[name]["type"] == "param":
                    trainable_params.add(model_trace.nodes[name]["value"])

            # grab guide parameters to train
            for name in guide_trace.nodes.keys():
                if guide_trace.nodes[name]["type"] == "param":
                    trainable_params.add(guide_trace.nodes[name]["value"])

        surrogate_loss = -surrogate_elbo
        surrogate_loss.backward()
        loss = -elbo

        pyro.get_param_store().mark_params_active(trainable_params)

        return loss
