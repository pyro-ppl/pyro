from __future__ import absolute_import, division, print_function

import torch


def _patch(target):
    parts = target.split('.')
    assert parts[0] == 'torch'
    module = torch
    for part in parts[1:-1]:
        module = getattr(module, part)
    name = parts[-1]
    old_fn = getattr(module, name)
    old_fn = getattr(old_fn, '_pyro_unpatched', old_fn)  # ensure patching is idempotent

    def decorator(new_fn):
        new_fn.__name__ = name
        new_fn._pyro_unpatched = old_fn
        setattr(module, name, new_fn)
        return new_fn

    return decorator


@_patch('torch._standard_gamma')
def _torch_standard_gamma(concentration):
    unpatched_fn = _torch_standard_gamma._pyro_unpatched
    if concentration.is_cuda:
        return unpatched_fn(concentration.cpu()).cuda(concentration.get_device())
    return unpatched_fn(concentration)


@_patch('torch.distributions.gamma._standard_gamma')
def _standard_gamma(concentration):
    if concentration.is_cuda:
        return concentration.cpu()._standard_gamma().cuda(concentration.get_device())
    return concentration._standard_gamma()


@_patch('torch._dirichlet_grad')
def _torch_dirichlet_grad(x, concentration, total):
    unpatched_fn = _torch_dirichlet_grad._pyro_unpatched
    if x.is_cuda:
        return unpatched_fn(x.cpu(), concentration.cpu(), total.cpu()).cuda(x.get_device())
    return unpatched_fn(x, concentration, total)


if torch.__version__.startswith('0.4.1'):

    # work around https://github.com/pytorch/pytorch/issues/9917
    @_patch('torch.bernoulli')
    def _torch_bernoulli(input, out=None):
        unpatched_fn = _torch_bernoulli._pyro_unpatched
        input = input.contiguous()
        return unpatched_fn(input) if out is None else unpatched_fn(input, out)

    # work around https://github.com/pytorch/pytorch/issues/9917
    @_patch('torch.poisson')
    def _torch_poisson(input):
        unpatched_fn = _torch_poisson._pyro_unpatched
        input = input.contiguous()
        return unpatched_fn(input)

    # work around https://github.com/pytorch/pytorch/issues/9521
    @_patch('torch._standard_gamma')  # noqa: F811
    def _torch_standard_gamma(concentration):
        concentration = concentration.contiguous()
        unpatched_fn = _torch_standard_gamma._pyro_unpatched
        if concentration.is_cuda:
            return unpatched_fn(concentration.cpu()).cuda(concentration.get_device())
        return unpatched_fn(concentration)

    # work around https://github.com/pytorch/pytorch/issues/9521
    @_patch('torch.distributions.gamma._standard_gamma')  # noqa: F811
    def _standard_gamma(concentration):
        concentration = concentration.contiguous()
        if concentration.is_cuda:
            return concentration.cpu()._standard_gamma().cuda(concentration.get_device())
        return concentration._standard_gamma()

    # work around https://github.com/pytorch/pytorch/issues/9521
    @_patch('torch._dirichlet_grad')  # noqa: F811
    def _torch_dirichlet_grad(x, concentration, total):
        unpatched_fn = _torch_dirichlet_grad._pyro_unpatched
        x = x.contiguous()
        concentration = concentration.contiguous()
        total = total.contiguous()
        if x.is_cuda:
            return unpatched_fn(x.cpu(), concentration.cpu(), total.cpu()).cuda(x.get_device())
        return unpatched_fn(x, concentration, total)


# these patches work after https://github.com/pytorch/pytorch/pull/10075
if hasattr(torch, 'broadcast_tensors'):

    # workaround lack of jit support for Categorical.log_prob()
    # this can be deleted after https://github.com/pytorch/pytorch/pull/10321
    @_patch('torch.distributions.categorical.Categorical.log_prob')
    def _log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)


__all__ = []
