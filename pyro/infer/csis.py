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
                 training_batch_size=10):
        super(CSIS, self).__init__(model, guide, num_inference_samples)
        self.model = model
        self.guide = guide
        self.optim = optim
        self.training_batch_size = training_batch_size
        self.validation_batch = None
        self.args = []
        self.kwargs = {}

    def set_validation_batch(self, validation_batch_size=20):
        """
        Samples a batch of model traces and stores it as an object property.
        """
        self.validation_batch = [self._sample_from_joint()
                                 for _ in range(validation_batch_size)]

    def set_args(self, *args, **kwargs):
        """
        Arguments are stored and passed to both the model and guide during
        training and inference.
        """
        self.args = args
        self.kwargs = kwargs

    def _traces(self, **kwargs):
        """
        Wrapping inherited _traces function to allow inference to be performed
        with the arguments set in set_args. Intended to allow user to specify
        keyword arguments containing observations when performing inference.
        """
        merged_kwargs = self.kwargs.copy()
        for k, v in kwargs.items():
            merged_kwargs[k] = v
        return super(CSIS, self)._traces(*self.args, **merged_kwargs)

    def step(self):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function. Any args or kwargs are
        passed to the model and guide.
        """
        loss = self.loss(grads=True)
        self.optim.step()
        self.optim.zero_grad()

        return torch_item(loss)

    def loss(self, grads=False, batch=None):
        """
        :returns: an estimate of the loss (expectation over p(x, y) of
            -log q(x, y) ) - where p is the model and q is the guide
        :rtype: float

        If a batch is provided, the loss is estimated using these traces
        Otherwise, a fresh batch is generated from the model.

        If grads is True, will also call `torch_backward` on loss.
        """
        if batch is None:
            batch = (self._sample_from_joint()
                     for _ in range(self.training_batch_size))
            batch_size = self.training_batch_size
        else:
            batch_size = len(batch)

        loss = 0
        for model_trace in batch:
            guide_trace = self._get_matched_trace(model_trace)
            particle_loss = -guide_trace.log_prob_sum() / batch_size
            if grads:
                torch_backward(particle_loss)
            loss += torch_item(particle_loss)

        warn_if_nan(loss, "loss")
        return loss

    def validation_loss(self):
        """
        :returns: loss estimated using validation batch
        :rtype: float

        Calculates loss on validation batch. `set_validation_batch` must have
        been called previously. Can be used to track loss in a less noisy way
        during training.
        """
        if self.validation_batch is None:
            raise ValueError("Validation batch not set.")

        return self.loss(grads=False, batch=self.validation_batch)

    def _get_matched_trace(self, model_trace):
        """
        :param model_trace: a trace from the model
        :type model_trace: pyro.poutine.trace_struct.Trace
        :returns: guide trace with sampled values matched to model_trace
        :rtype: pyro.poutine.trace_struct.Trace

        Returnss a guide trace with values at sample and observe statements
        matched to those in model_trace
        """
        updated_kwargs = self.kwargs
        for name in model_trace.observation_nodes:
            updated_kwargs[name] = model_trace.nodes[name]["value"]

        guide_trace = poutine.trace(poutine.replay(self.guide,
                                                   model_trace)
                                    ).get_trace(*self.args, **updated_kwargs)

        check_model_guide_match(model_trace, guide_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        return guide_trace

    def _sample_from_joint(self):
        """
        :returns: a sample from the joint distribution over unobserved and
            observed variables
        :rtype: pyro.poutine.trace_struct.Trace

        Returns a trace of the model without conditioning on any observations.
        """
        unconditioned_model = pyro.poutine.uncondition(self.model)
        return poutine.trace(unconditioned_model).get_trace(*self.args, **self.kwargs)
