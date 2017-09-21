from torch.autograd import Variable
import pyro
import pyro.poutine as poutine
from pyro.distributions.util import KLdiv


class KL_QP(object):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    :param optim: optimization function
    :param model_fixed: flag that controls whether the model parameters are fixed during optimization
    :param n_s_kldiv: number of kl-divergence samples if we sample it. At 1 uses guide-samples
    :param analytic: flag to detemrine if algorithm should look for analytical expression for kl div
    :type model_fixed: bool
    :param guide_fixed: flag that controls whether the guide parameters are fixed during optimization
    :type guide_fixed: bool

    This method performs variational inference by minimizing the
    `KL-divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
    between the actual and approximate posterior.

    Example::

        from pyro.infer.kl_qp import KL_QP

        kl_optim = KL_QP(model, guide,
                         pyro.optim(torch.optim.Adam, {"lr": .001}))
        for k in range(n_steps):
        # optimize
        kl_optim.step()
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 num_particles=1,
                 num_kl_samples=1,
                 analytic=True,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(KL_QP, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None
        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.num_particles = num_particles
        self.num_kl_samples = num_kl_samples
        self.analytic = analytic

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def populate_traces(self, *args, **kwargs):
        """
        Method to draw,store and evaluate multiple samples in traces
        """
        model_traces = []
        guide_traces = []
        log_r_per_sample = []

        for i in range(self.num_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r = model_trace.log_pdf() - guide_trace.log_pdf()
            log_r_per_sample.append(log_r)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)

        return model_traces, guide_traces, log_r_per_sample

    def eval_kld(self, m_site=None, g_site=None, analytic=False, num_kl_samples=1):
        """
        Method to evaluate the log-ratio
        If nr_sample=1, it reverts to just scoring the already sampled particles from the guide
        If, however, we want a better approximation to that local divergence, it will sample more particles
        from the local distributions or evaluate analytically
        :input n_s controls how often we would sample from the kl-divergence to evaluiate it, set to '10'.
        :input analytic determines if analytical forms for objectives are pursued
        :input m_site is a site of a model trace
        :input g_site is a site of a guide_trace

        """
        if analytic:
            kld_obj = KLdiv(dist_q=m_site['fn'], dist_p=g_site['fn'])
            return kld_obj.eval(analytical=analytic, num_kl_samples=num_kl_samples)
        else:
            if num_kl_samples == 1:
                return m_site["log_pdf"] - g_site["log_pdf"]
            else:
                kld_obj = KLdiv(dist_q=m_site['fn'], dist_p=g_site['fn'])
                return kld_obj.eval(analytical=analytic, num_kl_samples=num_kl_samples)

    def eval_objective(self, *args, **kwargs):
        """
        Evaluate Elbo by running num_particles often.
        Returns the Elbo as a value
        :parameter num_kl_samples :controls how often we would sample from the
                                   kl-divergence to evaluiate it, set to '10'.
        """
        model_traces = []
        guide_traces = []
        log_r_per_sample = []
        num_kl_samples = self.num_kl_samples
        analytic = self.analytic

        [model_traces, guide_traces, log_r_per_sample] = self.populate_traces(*args, **kwargs)

        elbo = 0.0
        for i in range(self.num_particles):
            elbo_particle = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    kld = self.eval_kld(m_site=guide_trace[name], g_site=model_trace[name],
                                        analytic=analytic, num_kl_samples=num_kl_samples)
                    elbo_particle -= kld.sum()
                else:
                    pass
            elbo += elbo_particle / self.num_particles
        loss = -elbo

        return loss.data[0]

    def eval_grad(self, *args, **kwargs):
        """
        Computes a surrogate loss, which, when differentiated yields an estimate of the gradient of the Elbo.
        Num_particle many samples are used to form the surrogate loss.
        :parameter num_kl_samples :controls how often we would sample from
                                   the kl-divergence to evaluiate it, set to '10'.
        """

        [model_traces, guide_traces, log_r_per_sample] = self.populate_traces(*args, **kwargs)
        num_kl_samples = self.num_kl_samples
        analytic = self.analytic

        elbo = 0.0
        for i in range(self.num_particles):
            elbo_particle = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            log_r = log_r_per_sample[i]

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        kld = self.eval_kld(m_site=guide_trace[name], g_site=model_trace[name],
                                            analytic=analytic, num_kl_samples=num_kl_samples)
                        elbo_particle -= kld.sum()
                    else:
                        elbo_particle += Variable(log_r.data) * guide_trace[name]["log_pdf"]
                else:
                    pass
            elbo += elbo_particle / self.num_particles

        # accumulate parameters
        all_trainable_params = []
        # get trace params from last model run
        if not self.model_fixed:
            for i in range(self.num_particles):
                model_trace = model_traces[i]
                for name in model_trace.keys():
                    if model_trace[name]["type"] == "param":
                        all_trainable_params.append(model_trace[name]["value"])
        # get trace params from last guide run
        if not self.guide_fixed:
            for i in range(self.num_particles):
                guide_trace = guide_traces[i]
                for name in guide_trace.keys():
                    if guide_trace[name]["type"] == "param":
                        all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))
        loss = -elbo

        return loss, all_trainable_params

    def step(self, *args, **kwargs):
        """
        Takes a single step of optimization by using the elbo loss and then applying autograd
        Returns the Elbo as a value
        If "loss" has been precomputed it can be passed into it, else it runs eval_grad with num_partricles
        """

        if 'loss_and_params' not in kwargs.keys():
            [loss, all_trainable_params] = self.eval_grad(*args, **kwargs)
        else:
            [loss, all_trainable_params] = kwargs['loss_and_params']

        loss.backward()

        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        pyro.util.zero_grads(all_trainable_params)

        return loss.data[0]
