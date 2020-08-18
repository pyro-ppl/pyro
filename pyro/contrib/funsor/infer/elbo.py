# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.infer import ELBO as OrigELBO


class ELBO(OrigELBO):

    def _get_trace(self, *args, **kwargs):
        raise ValueError("shouldn't be here!")

    def differentiable_loss(self, model, guide, *args, **kwargs):
        raise NotImplementedError("Must implement differentiable_loss")

    def loss(self, model, guide, *args, **kwargs):
        return self.differentiable_loss(model, guide, *args, **kwargs).detach().item()

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        loss.backward()
        return loss.item()
