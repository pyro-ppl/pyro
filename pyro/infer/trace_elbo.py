import pyro
import pyro.poutine as poutine


class Trace_ELBO(object):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    """
    def __init__(self,
                 model,
                 guide,
                 num_particles=1,
                 *args, **kwargs):
        # initialize
        super(Trace_ELBO, self).__init__()
        self.model = model
        self.guide = guide
        self.num_particles = num_particles

    def _get_traces(self, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """

        for i in range(self.num_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(self.model, guide_trace))(*args, **kwargs)
            log_r = model_trace.log_pdf() - guide_trace.log_pdf()
            yield model_trace, guide_trace, log_r

    def loss(self, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        :returns: returns an estimate of the ELBO
        :rtype: float
        """
        elbo = 0.0
        for model_trace, guide_trace, log_r in self._get_traces(*args, **kwargs):
            elbo_particle = 0.0

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    elbo_particle += model_trace[name]["log_pdf"]
                    elbo_particle -= guide_trace[name]["log_pdf"]

            elbo += elbo_particle / self.num_particles

        loss = -elbo
        return loss.data[0]

    def loss_and_grads(self, *args, **kwargs):
        """
        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        :returns: returns an estimate of the ELBO
        :rtype: torch.autograd.Variable
        """
        elbo = 0.0
        surrogate_elbo = 0.0
        trainable_params = set()

        # grab a trace from the generator
        for model_trace, guide_trace, log_r in self._get_traces(*args, **kwargs):
            elbo_particle = 0.0
            surrogate_elbo_particle = 0.0

            # compute elbo and surrogate elbo
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                    surrogate_elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    lp_lq = model_trace[name]["log_pdf"] - guide_trace[name]["log_pdf"]
                    elbo_particle += lp_lq
                    if model_trace[name]["fn"].reparameterized:
                        surrogate_elbo_particle += lp_lq
                    else:
                        surrogate_elbo_particle += model_trace[name]["log_pdf"] + \
                            log_r.detach() * guide_trace[name]["log_pdf"]

            elbo += elbo_particle / self.num_particles
            surrogate_elbo += surrogate_elbo_particle / self.num_particles

            # grab model parameters to train
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    trainable_params.add(model_trace[name]["value"])

            # grab guide parameters to train
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    trainable_params.add(guide_trace[name]["value"])

        surrogate_loss = -surrogate_elbo
        surrogate_loss.backward()
        loss = -elbo

        pyro.get_param_store().mark_params_active(trainable_params)

        return loss
