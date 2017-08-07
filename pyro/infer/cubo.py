import six
import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
import pyro.poutine as poutine
from pyro.infer.abstract_infer import AbstractInfer

import pdb as pdb

def zero_grads(tensors):
    """
    Sets gradients of list of Variables to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            if p.grad.volatile:
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


class CUBO_logc(AbstractInfer):
    """
    Cubo with correction term
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 n_cubo=2,
                 nr_particles = 3,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(CUBO, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.n_cubo = n_cubo
        self.nr_particles = nr_particles

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        model_traces = []
        guide_traces = []
        log_weights = []
        for i in range(self.nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r_raw = model_trace.batch_log_pdf() - guide_trace.batch_log_pdf()
            log_weights.append(log_r_raw)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)

        log_weights_tensor = torch.stack(log_weights,1)

        log_r_max = Variable(torch.max(log_weights_tensor,1)[0].data)
        log_r = Variable((log_weights_tensor - log_r_max.expand_as(log_weights_tensor)).data)

        w_n = Variable(torch.exp(log_r * self.n_cubo).data)

        w_0 = Variable(torch.exp(log_r).data)
        w_0_sum = w_0.sum(1)
        w_0_norm = w_0 / w_0_sum.expand_as(w_0)
        w_0n = torch.pow(w_0_norm,self.n_cubo)


        cubo = 0.0
        exp_cubo = 0.0
        grad_cubo = 0.0
        for i in range(self.nr_particles):
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    log_r_s += model_trace[name]["batch_log_pdf"]# * (self.n_cubo) / self.nr_particles
                    pass
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        # print "name",model_trace[name]
                        log_r_s += log_r[:,i] * model_trace[name]["batch_log_pdf"]# * (self.n_cubo) / self.nr_particles
                        log_r_s -= log_r[:,i] * guide_trace[name]["batch_log_pdf"]# * (self.n_cubo) / self.nr_particles

                    else:
                        log_r_s += log_r[:,i] * guide_trace[name]["batch_log_pdf"]#* w_n[:,i] * (1-self.n_cubo) / self.nr_particles

                else:
                    pass

            #pdb.set_trace()
            exp_cubo += ( torch.exp(log_r_s * self.n_cubo) )# / self.nr_particles
            #exp_cubo += ( torch.exp( (log_r_s - log_r_max) * self.n_cubo) ) / self.nr_particles
            #exp_cubo += w_n[:,i] * log_r_s /self.nr_particles
            #exp_cubo += w_n[:,i] * log_r_s /self.nr_particles
            #torch.exp( (log_r_s - Variable(log_r_max.data)) * self.n_cubo) / self.nr_particles
            grad_cubo += log_r_s

        exp_cubo_sum = exp_cubo.sum()

        #unimportant quantity, only used for reporting, not for gradient
        cubo = (torch.log(exp_cubo)/self.n_cubo ).sum()

        # accumulate parameters
        all_trainable_params = []
        # get trace params from last model run
        if not self.model_fixed:
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    all_trainable_params.append(model_trace[name]["value"])
        # get trace params from last guide run
        if not self.guide_fixed:
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # gradients
        loss = exp_cubo_sum
        loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        # return the log transform of the expectation
        return cubo.data[0]

class CUBO_logcn(AbstractInfer):
    """
    Cubo with normalized correction
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 n_cubo=2,
                 nr_particles = 3,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(CUBO_logcn, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.n_cubo = n_cubo
        self.nr_particles = nr_particles

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        model_traces = []
        guide_traces = []
        log_weights = []
        for i in range(self.nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r_raw = model_trace.batch_log_pdf() - guide_trace.batch_log_pdf()
            log_weights.append(log_r_raw)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)


        log_weights_tensor = torch.stack(log_weights,1)

        log_r_max = Variable(torch.max(log_weights_tensor,1)[0].data)
        log_r = Variable((log_weights_tensor - log_r_max.expand_as(log_weights_tensor)).data)

        w_n = Variable(torch.exp(log_r * self.n_cubo).data)

        w_0 = Variable(torch.exp(log_r).data)
        w_0_sum = w_0.sum(1)
        w_0_norm = w_0 / w_0_sum.expand_as(w_0)
        w_0n = torch.pow(w_0_norm,self.n_cubo)


        cubo = 0.0
        exp_cubo = 0.0
        grad_cubo = 0.0
        for i in range(self.nr_particles):
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    #log_r_s += model_trace[name]["batch_log_pdf"]# * (self.n_cubo) / self.nr_particles
                    pass
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        # print "name",model_trace[name]
                        #log_r_s += model_trace[name]["batch_log_pdf"]* w_n[:,i]# * (self.n_cubo) / self.nr_particles
                        #log_r_s -= guide_trace[name]["batch_log_pdf"]* w_n[:,i]# * (self.n_cubo) / self.nr_particles

                        log_r_s += w_n[:,i] * guide_trace[name]["batch_log_pdf"] * (1-self.n_cubo) / self.nr_particles

                    else:
                        log_r_s += w_n[:,i] * guide_trace[name]["batch_log_pdf"] * (1-self.n_cubo) / self.nr_particles

                else:
                    pass

            #pdb.set_trace()
            exp_cubo += ( torch.exp(log_r_s * self.n_cubo) )# / self.nr_particles
            #exp_cubo += ( torch.exp( (log_r_s - log_r_max) * self.n_cubo) ) / self.nr_particles
            #exp_cubo += w_n[:,i] * log_r_s /self.nr_particles
            #exp_cubo += w_n[:,i] * log_r_s /self.nr_particles
            #torch.exp( (log_r_s - Variable(log_r_max.data)) * self.n_cubo) / self.nr_particles
            grad_cubo += log_r_s

        exp_cubo_sum = grad_cubo.sum()

        #unimportant quantity, only used for reporting, not for gradient
        cubo = (torch.log(exp_cubo)/self.n_cubo ).sum()

        # accumulate parameters
        all_trainable_params = []
        # get trace params from last model run
        if not self.model_fixed:
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    all_trainable_params.append(model_trace[name]["value"])
        # get trace params from last guide run
        if not self.guide_fixed:
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # gradients
        loss = exp_cubo_sum
        loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        # return the log transform of the expectation
        return cubo.data[0]

class CUBO(AbstractInfer):
    """
    A new, Trace and Poutine-based implementation of SVI
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 n_cubo=2,
                 nr_particles = 3,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(CUBO, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.n_cubo = n_cubo
        self.nr_particles = nr_particles

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        model_traces = []
        guide_traces = []
        log_weights = []
        for i in range(self.nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r_raw = model_trace.batch_log_pdf() - guide_trace.batch_log_pdf()
            log_weights.append(log_r_raw)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)


        # log_weights_tensor = Variable(torch.stack(log_weights,1).data)

        # log_r_max = Variable(torch.max(log_weights_tensor,1)[0].data)
        # log_r = Variable((log_weights_tensor - log_r_max.expand_as(log_weights_tensor)).data)

        # w_n = Variable(torch.exp(log_r * self.n_cubo).data)

        # w_0 = Variable(torch.exp(log_r).data)
        # w_0_sum = w_0.sum(1)
        # w_0_norm = w_0 / w_0_sum.expand_as(w_0)
        # w_0n = torch.pow(w_0_norm,self.n_cubo)


        cubo = 0.0
        exp_cubo = 0.0
        for i in range(self.nr_particles):
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    log_r_s += model_trace[name]["batch_log_pdf"]
                    pass
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        # print "name",model_trace[name]
                        log_r_s += model_trace[name]["batch_log_pdf"]
                        log_r_s -= guide_trace[name]["batch_log_pdf"]

                    else:
                        log_r_s += model_trace[name]["batch_log_pdf"]
                        log_r_s -= guide_trace[name]["batch_log_pdf"]
                else:
                    pass

            # just the raw objective
            exp_cubo += ( torch.exp(log_r_s * self.n_cubo) ) / self.nr_particles

        exp_cubo_sum = exp_cubo.sum()
        cubo = (torch.log(exp_cubo)/self.n_cubo ).sum()

        #print(cubo.data.cpu().numpy())[0]
        #pdb.set_trace()
        # accumulate parameters
        all_trainable_params = []
        # get trace params from last model run
        if not self.model_fixed:
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    all_trainable_params.append(model_trace[name]["value"])
        # get trace params from last guide run
        if not self.guide_fixed:
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # gradients
        loss = exp_cubo_sum
        loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        # return the log transform of the expectation
        return cubo.data[0]


class CUBO_grad(AbstractInfer):
    """
    Implements the gradient of Cubo in normal form
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 n_cubo=2,
                 nr_particles = 3,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(CUBO_grad, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.n_cubo = n_cubo
        self.nr_particles = nr_particles

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        model_traces = []
        guide_traces = []
        log_weights = []
        for i in range(self.nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r_raw = model_trace.batch_log_pdf() - guide_trace.batch_log_pdf()
            log_weights.append(log_r_raw)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)


        log_weights_tensor = Variable(torch.stack(log_weights,1).data)

        log_r_max = Variable(torch.max(log_weights_tensor,1)[0].data)
        log_w_shifted = Variable((log_weights_tensor - log_r_max.expand_as(log_weights_tensor)).data)

        w_n = Variable((log_weights_tensor*self.n_cubo).data)
        #w_n = Variable(torch.exp(log_w_shifted*self.n_cubo).data)
        #w_n = torch.pow(Variable(torch.exp(log_w_shifted).data),self.n_cubo)

        w_0 = Variable(torch.exp(log_w_shifted).data)
        w_0_sum = w_0.sum(1)
        w_0_norm = w_0 / w_0_sum.expand_as(w_0)
        w_0n = torch.pow(w_0_norm,self.n_cubo)

        cubo = 0.0
        exp_cubo = 0.0


        non_rep = False
        for i in range(self.nr_particles):
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    log_r_s += model_trace[name]["batch_log_pdf"]
                    pass
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        #print(model_trace[name]["fn"].reparameterized)
                        log_r_s += model_trace[name]["batch_log_pdf"]
                        log_r_s -= guide_trace[name]["batch_log_pdf"]

                    else:
                        non_rep = True
                        #pdb.set_trace()
                        pass
                else:
                    pass

            if non_rep:
                #pdb.set_trace()
                log_r_s = 0.0
                for name in model_trace.keys():
                    if model_trace[name]["type"] == "observe":
                        #log_r_s += model_trace[name]["batch_log_pdf"]
                        pass

                    elif model_trace[name]["type"] == "sample":
                        log_r_s += guide_trace[name]["batch_log_pdf"]

            # just the raw objective
            if non_rep:
                exp_cubo += w_n[:,i] * log_r_s * (1-self.n_cubo) / self.nr_particles
            else:
                exp_cubo += w_n[:,i] * log_r_s * self.n_cubo / self.nr_particles

        exp_cubo_sum = exp_cubo.sum()
        cubo = (torch.log(exp_cubo)/self.n_cubo ).sum()

        # pdb.set_trace()

        # accumulate parameters
        all_trainable_params = []
        # get trace params from last model run
        if not self.model_fixed:
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    all_trainable_params.append(model_trace[name]["value"])
        # get trace params from last guide run
        if not self.guide_fixed:
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # gradients
        loss = exp_cubo_sum
        loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        # return the log transform of the expectation
        return exp_cubo_sum.data[0]


class CUBO_massive(AbstractInfer):
    """
    A new, Trace and Poutine-based implementation of SVI
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 n_cubo=2,
                 nr_particles = 3,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(CUBO_massive, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.n_cubo = n_cubo
        self.nr_particles = nr_particles

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        model_traces = []
        guide_traces = []
        log_weights = []

        # pdb.set_trace()

        for i in range(self.nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)

            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "sample":
                    samples_0 = guide_trace[name]['value'][0,:].expand_as(guide_trace[name]['value'][0,:])
                    guide_trace[name]['value'][:] = samples_0

            model_trace = poutine.trace(poutine.replay(self.model, guide_trace))(*args, **kwargs)
            # for name in model_trace.keys():
            #     if model_trace[name]["type"] == "observe":
            #         avr_lk += model_trace[name]['batch_log_pdf']
            log_r_raw = model_trace.batch_log_pdf() - guide_trace.batch_log_pdf()
            log_weights.append(log_r_raw)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)

        log_weights_tensor = Variable(torch.stack(log_weights,1).data)

        # pdb.set_trace()

        log_r_max = Variable(torch.max(log_weights_tensor,1)[0].data)
        log_w_shifted = Variable((log_weights_tensor - log_r_max.expand_as(log_weights_tensor)).data)

        w_n = Variable((log_weights_tensor*self.n_cubo).data)
        #w_n = Variable(torch.exp(log_w_shifted*self.n_cubo).data)
        #w_n = torch.pow(Variable(torch.exp(log_w_shifted).data),self.n_cubo)

        w_0 = Variable(torch.exp(log_w_shifted).data)
        w_0_sum = w_0.sum(1)
        w_0_norm = w_0 / w_0_sum.expand_as(w_0)
        w_0n = torch.pow(w_0_norm,self.n_cubo)

        cubo = 0.0
        exp_cubo = 0.0

        avr_lk = 0.0
        non_rep = False
        for i in range(self.nr_particles):
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    avr_lk += model_trace[name]["batch_log_pdf"] / self.nr_particles
                    pass
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        #print(model_trace[name]["fn"].reparameterized)
                        log_r_s += model_trace[name]["batch_log_pdf"]
                        log_r_s -= guide_trace[name]["batch_log_pdf"]

                    else:
                        non_rep = True
                        #pdb.set_trace()
                        pass
                else:
                    pass

            if non_rep:
                #pdb.set_trace()
                log_r_s = 0.0
                for name in model_trace.keys():
                    if model_trace[name]["type"] == "observe":
                        #log_r_s += model_trace[name]["batch_log_pdf"]
                        pass

                    elif model_trace[name]["type"] == "sample":
                        log_r_s += guide_trace[name]["batch_log_pdf"]

            # just the raw objective
            if non_rep:
                exp_cubo += w_n[:,i] * log_r_s * (1-self.n_cubo) / self.nr_particles
            else:
                exp_cubo += w_n[:,i] * log_r_s * self.n_cubo / self.nr_particles

        exp_cubo_sum = exp_cubo.sum()
        cubo = (torch.log(exp_cubo)/self.n_cubo ).sum()

        # accumulate parameters
        all_trainable_params = []
        # get trace params from last model run
        if not self.model_fixed:
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    all_trainable_params.append(model_trace[name]["value"])
        # get trace params from last guide run
        if not self.guide_fixed:
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # gradients
        loss = exp_cubo_sum
        loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        # return the log transform of the expectation
        return exp_cubo_sum.data[0]
