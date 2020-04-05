# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.importance import Importance
from pyro.infer.util import torch_item
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, warn_if_nan


class CSIS(Importance):
    """
    Compiled Sequential Importance Sampling, allowing compilation of a guide
    program to minimise KL(model posterior || guide), and inference with
    importance sampling.

    **Reference**
    "Inference Compilation and Universal Probabilistic Programming" `pdf https://arxiv.org/pdf/1610.09900.pdf`

    :param model: probabilistic model defined as a function. Must accept a
        keyword argument named `observations`, in which observed values are
        passed as, with the names of nodes as the keys.
    :param guide: guide function which is used as an approximate posterior. Must
        also accept `observations` as keyword argument.
    :param optim: a Pyro optimizer
    :type optim: pyro.optim.PyroOptim
    :param num_inference_samples: The number of importance-weighted samples to
        draw during inference.
    :param training_batch_size: Number of samples to use to approximate the loss
        before each gradient descent step during training.
    :param validation_batch_size: Number of samples to use for calculating
        validation loss (will only be used if `.validation_loss` is called).
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 num_inference_samples=10,
                 training_batch_size=10,
                 validation_batch_size=20):
        super().__init__(model, guide, num_inference_samples)
        self.model = model
        self.guide = guide
        self.optim = optim
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_batch = None

    def set_validation_batch(self, *args, **kwargs):
        """
        Samples a batch of model traces and stores it as an object property.

        Arguments are passed directly to model.
        """
        self.validation_batch = [self._sample_from_joint(*args, **kwargs)
                                 for _ in range(self.validation_batch_size)]

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function. Arguments are passed to the
        model and guide.
        """
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(True, None, *args, **kwargs)

        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values()
                     if site["value"].grad is not None)

        self.optim(params)

        pyro.infer.util.zero_grads(params)

        return torch_item(loss)

    def loss_and_grads(self, grads, batch, *args, **kwargs):
        """
        :returns: an estimate of the loss (expectation over p(x, y) of
            -log q(x, y) ) - where p is the model and q is the guide
        :rtype: float

        If a batch is provided, the loss is estimated using these traces
        Otherwise, a fresh batch is generated from the model.

        If grads is True, will also call `backward` on loss.

        `args` and `kwargs` are passed to the model and guide.
        """
        if batch is None:
            batch = (self._sample_from_joint(*args, **kwargs)
                     for _ in range(self.training_batch_size))
            batch_size = self.training_batch_size
        else:
            batch_size = len(batch)

        loss = 0
        for model_trace in batch:
            with poutine.trace(param_only=True) as particle_param_capture:
                guide_trace = self._get_matched_trace(model_trace, *args, **kwargs)
            particle_loss = self._differentiable_loss_particle(guide_trace)
            particle_loss /= batch_size

            if grads:
                guide_params = set(site["value"].unconstrained()
                                   for site in particle_param_capture.trace.nodes.values())
                guide_grads = torch.autograd.grad(particle_loss, guide_params, allow_unused=True)
                for guide_grad, guide_param in zip(guide_grads, guide_params):
                    guide_param.grad = guide_grad if guide_param.grad is None else guide_param.grad + guide_grad

            loss += torch_item(particle_loss)

        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, guide_trace):
        return -guide_trace.log_prob_sum()

    def validation_loss(self, *args, **kwargs):
        """
        :returns: loss estimated using validation batch
        :rtype: float

        Calculates loss on validation batch. If no validation batch is set,
        will set one by calling `set_validation_batch`. Can be used to track
        the loss in a less noisy way during training.

        Arguments are passed to the model and guide.
        """
        if self.validation_batch is None:
            self.set_validation_batch(*args, **kwargs)

        return self.loss_and_grads(False, self.validation_batch, *args, **kwargs)

    def _get_matched_trace(self, model_trace, *args, **kwargs):
        """
        :param model_trace: a trace from the model
        :type model_trace: pyro.poutine.trace_struct.Trace
        :returns: guide trace with sampled values matched to model_trace
        :rtype: pyro.poutine.trace_struct.Trace

        Returns a guide trace with values at sample and observe statements
        matched to those in model_trace.

        `args` and `kwargs` are passed to the guide.
        """
        kwargs["observations"] = {}
        for node in itertools.chain(model_trace.stochastic_nodes, model_trace.observation_nodes):
            if "was_observed" in model_trace.nodes[node]["infer"]:
                model_trace.nodes[node]["is_observed"] = True
                kwargs["observations"][node] = model_trace.nodes[node]["value"]

        guide_trace = poutine.trace(poutine.replay(self.guide,
                                                   model_trace)
                                    ).get_trace(*args, **kwargs)

        check_model_guide_match(model_trace, guide_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        return guide_trace

    def _sample_from_joint(self, *args, **kwargs):
        """
        :returns: a sample from the joint distribution over unobserved and
            observed variables
        :rtype: pyro.poutine.trace_struct.Trace

        Returns a trace of the model without conditioning on any observations.

        Arguments are passed directly to the model.
        """
        unconditioned_model = pyro.poutine.uncondition(self.model)
        return poutine.trace(unconditioned_model).get_trace(*args, **kwargs)
