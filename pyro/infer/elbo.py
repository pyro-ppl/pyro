from .trace_elbo import Trace_ELBO
from .tracegraph_elbo import TraceGraph_ELBO


class ELBO(object):
    def __init__(self,
                 model,
                 guide,
                 num_particles=1,
                 trace_graph=False,
                 *args, **kwargs):
        # initialize
        super(ELBO, self).__init__()
        self.model = model
        self.guide = guide
        self.num_particles = num_particles
        self.trace_graph = trace_graph
        if self.trace_graph:
            self.which_elbo = TraceGraph_ELBO(model, guide, num_particles=num_particles, *args, **kwargs)
        else:
            self.which_elbo = Trace_ELBO(model, guide, num_particles=num_particles, *args, **kwargs)

    def loss(self, *args, **kwargs):
        """
        Evaluate Elbo by running num_particles often.
        Returns the Elbo as a value
        """
        return self.which_elbo.loss(*args, **kwargs)

    def loss_and_grads(self, *args, **kwargs):
        """
        computes the elbo as well as the surrogate elbo. performs backward on latter.
        num_particle many samples are used to form the estimators.
        returns an estimate of the elbo as well as the trainable_params_dict.
        implicitly returns gradients via param.grad for each param in the trainable_params_dict.
        if trace_graph = True, we also return the baseline loss and baseline params
        """
        return self.which_elbo.loss_and_grads(*args, **kwargs)
