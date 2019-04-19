from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints, kl_divergence, register_kl

from pyro.distributions.torch_distribution import IndependentConstraint, TorchDistributionMixin
from pyro.distributions.util import sum_rightmost


# This overloads .log_prob() and .enumerate_support() to speed up evaluating
# log_prob on the support of this variable: we can completely avoid tensor ops
# and merely reshape the self.logits tensor. This is especially important for
# Pyro models that use enumeration.
class Categorical(torch.distributions.Categorical, TorchDistributionMixin):

    def log_prob(self, value):
        if getattr(value, '_pyro_categorical_support', None) == id(self):
            # Assume value is a reshaped torch.arange(event_shape[0]).
            # In this case we can call .reshape() rather than torch.gather().
            if not torch._C._get_tracing_state():
                if self._validate_args:
                    self._validate_sample(value)
                assert value.size(0) == self.logits.size(-1)
            logits = self.logits
            if logits.dim() <= value.dim():
                logits = logits.reshape((1,) * (1 + value.dim() - logits.dim()) + logits.shape)
            if not torch._C._get_tracing_state():
                assert logits.size(-1 - value.dim()) == 1
            return logits.transpose(-1 - value.dim(), -1).squeeze(-1)
        return super(Categorical, self).log_prob(value)

    def enumerate_support(self, expand=True):
        result = super(Categorical, self).enumerate_support(expand=expand)
        if not expand:
            result._pyro_categorical_support = id(self)
        return result


class MultivariateNormal(torch.distributions.MultivariateNormal, TorchDistributionMixin):
    support = IndependentConstraint(constraints.real, 1)  # TODO move upstream


class Independent(torch.distributions.Independent, TorchDistributionMixin):
    @constraints.dependent_property
    def support(self):
        return IndependentConstraint(self.base_dist.support, self.reinterpreted_batch_ndims)

    @property
    def _validate_args(self):
        return self.base_dist._validate_args

    @_validate_args.setter
    def _validate_args(self, value):
        self.base_dist._validate_args = value


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    kl = kl_divergence(p.base_dist, q.base_dist)
    if p.reinterpreted_batch_ndims:
        kl = sum_rightmost(kl, p.reinterpreted_batch_ndims)
    return kl


# Programmatically load all distributions from PyTorch.
__all__ = []
for _name, _Dist in torch.distributions.__dict__.items():
    if not isinstance(_Dist, type):
        continue
    if not issubclass(_Dist, torch.distributions.Distribution):
        continue
    if _Dist is torch.distributions.Distribution:
        continue

    try:
        _PyroDist = locals()[_name]
    except KeyError:
        _PyroDist = type(_name, (_Dist, TorchDistributionMixin), {})
        _PyroDist.__module__ = __name__
        locals()[_name] = _PyroDist

    _PyroDist.__doc__ = '''
    Wraps :class:`{}.{}` with
    :class:`~pyro.distributions.torch_distribution.TorchDistributionMixin`.
    '''.format(_Dist.__module__, _Dist.__name__)

    __all__.append(_name)


# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: pyro.distributions.{0}
    '''.format(_name)
    for _name in sorted(__all__)
])
