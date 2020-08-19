# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import weakref

import pyro.ops.jit
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


def make_jit_elbo(elbo_cls):

    def differentiable_loss(self, model, guide, *args, **kwargs):
        kwargs['_model_id'] = id(model)
        kwargs['_guide_id'] = id(guide)
        if getattr(self, '_differentiable_loss', None) is None:
            # build a closure for differentiable_loss
            weakself = weakref.ref(self)

            @pyro.ops.jit.trace(ignore_warnings=self.ignore_jit_warnings,
                                jit_options=self.jit_options)
            def differentiable_loss(*args, **kwargs):
                kwargs.pop('_model_id')
                kwargs.pop('_guide_id')
                self = weakself()
                return super(type(self), self).differentiable_loss(model, guide, *args, **kwargs)

            self._differentiable_loss = differentiable_loss

        return self._differentiable_loss(*args, **kwargs)

    assert issubclass(elbo_cls, ELBO)
    return type("Jit" + elbo_cls.__name__, (elbo_cls,), {"differentiable_loss": differentiable_loss})
