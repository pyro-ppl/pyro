from __future__ import absolute_import, division, print_function

import pyro
import pyro.poutine as poutine
from pyro.infer.importance import Importance
from pyro.infer.util import torch_backward, torch_item
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, warn_if_nan


class CSIS(Importance):
    """
    Compiled Sequential Importance Sampling, allowing compilation of a guide
    program to minimise KL(model posterior || guide), and inference with
    importance sampling.

    **Reference**
    "Inference Compilation and Universal Probabilistic Programming" `pdf https://arxiv.org/pdf/1610.09900.pdf`

    :param model: probabilistic model defined as a function. Must accept
        keyword arguments with names of observed sites in model.
    :param guide: guide function which is used as an approximate posterior. Must
        accept keyword arguments with names of observed sites in model.
    :param optim: a PyTorch optimizer
    :type optim: torch.optim.Optimizer
    :param num_inference_samples: The number of importance-weighted samples to
        draw during inference.
    :param training_batch_size: Number of samples to use to approximate the loss
        before each gradient descent step during training.
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 num_inference_samples=10,
                 training_batch_size=10,
                 validation_batch_size=20):
        super(CSIS, self).__init__(model, guide, num_inference_samples)
        self.model = model
        self.guide = guide
        self.optim = optim
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_batch = None

    def set_validation_batch(self, *args, **kwargs):
        """
        Samples a batch of model traces and stores it as an object property.
        """
        self.validation_batch = [self._sample_from_joint(*args, **kwargs)
                                 for _ in range(self.validation_batch_size)]

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function. Any args or kwargs are
        passed to the model and guide.
        """
        loss = self.loss(True, None, *args, **kwargs)
        self.optim.step()
        self.optim.zero_grad()

        return torch_item(loss)

    def loss(self, grads, batch, *args, **kwargs):
        """
        :returns: an estimate of the loss (expectation over p(x, y) of
            -log q(x, y) ) - where p is the model and q is the guide
        :rtype: float

        If a batch is provided, the loss is estimated using these traces
        Otherwise, a fresh batch is generated from the model.

        If grads is True, will also call `torch_backward` on loss.
        """
        if batch is None:
            batch = (self._sample_from_joint(*args, **kwargs)
                     for _ in range(self.training_batch_size))
            batch_size = self.training_batch_size
        else:
            batch_size = len(batch)

        loss = 0
        for model_trace in batch:
            guide_trace = self._get_matched_trace(model_trace, *args, **kwargs)
            particle_loss = -guide_trace.log_prob_sum() / batch_size
            if grads:
                torch_backward(particle_loss)
            loss += torch_item(particle_loss)

        warn_if_nan(loss, "loss")
        return loss

    def validation_loss(self, *args, **kwargs):
        """
        :returns: loss estimated using validation batch
        :rtype: float

        Calculates loss on validation batch. `set_validation_batch` must have
        been called previously. Can be used to track loss in a less noisy way
        during training.
        """
        if self.validation_batch is None:
            self.set_validation_batch(*args, **kwargs)

        return self.loss(grads=False, batch=self.validation_batch, *args, **kwargs)

    def _get_matched_trace(self, model_trace, *args, **kwargs):
        """
        :param model_trace: a trace from the model
        :type model_trace: pyro.poutine.trace_struct.Trace
        :returns: guide trace with sampled values matched to model_trace
        :rtype: pyro.poutine.trace_struct.Trace

        Returnss a guide trace with values at sample and observe statements
        matched to those in model_trace
        """
        updated_kwargs = kwargs
        for name in model_trace.observation_nodes:
            updated_kwargs[name] = model_trace.nodes[name]["value"]

        guide_trace = poutine.trace(poutine.replay(self.guide,
                                                   model_trace)
                                    ).get_trace(*args, **updated_kwargs)

        check_model_guide_match(model_trace, guide_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        return guide_trace

    def _sample_from_joint(self, *args, **kwargs):
        """
        :returns: a sample from the joint distribution over unobserved and
            observed variables
        :rtype: pyro.poutine.trace_struct.Trace

        Returns a trace of the model without conditioning on any observations.
        """
        unconditioned_model = pyro.poutine.uncondition(self.model)
        return poutine.trace(unconditioned_model).get_trace(*args, **kwargs)
