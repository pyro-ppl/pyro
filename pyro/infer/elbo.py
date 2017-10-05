from .trace_elbo import Trace_ELBO
from .tracegraph_elbo import TraceGraph_ELBO


class ELBO(object):
    """
    :param model: the model (callable)
    :param guide: the guide (callable), i.e. the variational distribution
    :param num_particles: the number of particles (samples) used to form the estimator.
    :param trace_graph: boolean. whether to keep track of dependency information when running the
	model and guide. this information can be used to form a gradient estimator with lower variance
	in the case that some of the random variables are non-reparameterized.
	note: for a model with many random variables, keeping track of the dependency information
	can be expensive.
    Note: ELBO dispatches to Trace_ELBO and TraceGraph_ELBO
    """
    def __init__(self,
                 model,
                 guide,
                 num_particles=1,
                 trace_graph=False):
        super(ELBO, self).__init__()
        self.model = model
        self.guide = guide
        self.num_particles = num_particles
        self.trace_graph = trace_graph
        if self.trace_graph:
            self.which_elbo = TraceGraph_ELBO(model, guide, num_particles=num_particles)
        else:
            self.which_elbo = Trace_ELBO(model, guide, num_particles=num_particles)

    def loss(self, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        Returns the ELBO as a float.
        """
        return self.which_elbo.loss(*args, **kwargs)

    def loss_and_grads(self, *args, **kwargs):
        """
        - computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
	- performs backward on the latter.
        - num_particle many samples are used to form the estimators.
        - returns an estimate of the ELBO as well as the trainable_params_dict.
        - if trace_graph is True, we also return the baseline loss and baseline params
        - implicitly returns gradients via param.grad for each param in the trainable_params_dict.
        """
        return self.which_elbo.loss_and_grads(*args, **kwargs)
