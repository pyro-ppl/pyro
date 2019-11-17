from collections import OrderedDict

import torch

import pyro
import pyrp.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item
from pyro.util import warn_if_nan


class Trace_CRPS:
    """
    Posterior predictive CRPS loss.

    This is a non-Bayesian, likelihood-free method; no densities are evaluated.
    """
    def __init__(self,
                 num_particles=2,
                 max_plate_nesting=float('inf')):
        assert num_particles >= 2
        self.max_plate_nesting = max_plate_nesting
        self.vectorize_particles = True

    def _get_samples(self, model, guide, *args, **kwargs):
        if self._max_plate_nesting == float("inf"):
            ELBO._guess_max_plate_nesting(self, model, guide, args, **kwargs)
        vectorize = pyro.plate("particles", self.num_particles, dim=-1 - self.max_plate_nesting)

        # Trace the guide as in ELBO.
        with poutine.trace() as tr, vectorize:
            guide(*args, **kwargs)
        guide_trace = tr.trace

        # Trace the model, saving obs in tr2 and posterior predictives in tr1.
        with poutine.trace() as tr1, poutine.uncondition():
            with poutine.trace() as tr2:
                with poutine.replay(trace=guide_trace), vectorize:
                    model(*args, **kwargs)

        # Extract observations and posterior predictive samples.
        data = OrderedDict()
        samples = OrderedDict()
        for name, site in tr2.trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                data[name] = site["value"]
                samples[name] = tr1.trace.nodes[name]["value"]
        assert set(data) == set(samples)

        return data, samples

    def differentiable_loss(self, model, guide, *args, **kwargs):
        data, samples = self._get_traces()

        # Compute mean average error and generalized entropy.
        squared_error = 0  # E[ (X - x)^2 ]
        squared_entropy = 0  # E[ (X - X')^2 ]
        prototype = next(iter(data.values()))
        i = prototype.new_ones(self.num_particles, self.num_particles).tril(-1).nonzero()
        for name, obs in data.items():
            obs = obs.reshape(-1)
            sample = samples[name].reshape(self.num_samples, -1)
            squared_error = squared_error + (sample - obs).pow(2).sum(-1)
            x, y = sample[i].unbind(0)
            squared_entropy = squared_entropy + (x - y).pow(2).sum(-1)

        error = squared_error.sqrt().mean()  # E[ |X-x| ]
        entropy = squared_entropy.sqrt().mean()  # E[ |X-X'| ]
        loss = error - 0.5 * entropy
        return loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the CRPS
        :rtype: float

        Evaluates the CRPS with an estimator that uses num_particles many samples/particles.
        """
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        return torch_item(loss)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the CRPS
        :rtype: float

        Computes the CRPS as well as the surrogate CRPS that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        warn_if_nan(loss, "loss")
        loss.backward()
        return torch_item(loss)
