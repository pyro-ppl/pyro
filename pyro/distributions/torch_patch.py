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


if torch.__version__ >= '0.4.1':

    # work around https://github.com/pytorch/pytorch/issues/10241
    # this can be deleted after https://github.com/pytorch/pytorch/pull/10269
    @_patch('torch.log')
    def _torch_log(input, out=None):
        unpatched_fn = _torch_log._pyro_unpatched
        input = input.contiguous()
        return unpatched_fn(input) if out is None else unpatched_fn(input, out)

    # work around https://github.com/pytorch/pytorch/issues/10241
    # this can be deleted after https://github.com/pytorch/pytorch/pull/10269
    @_patch('torch.Tensor.log')
    def _Tensor_log(self):
        unpatched_fn = _Tensor_log._pyro_unpatched
        self = self.contiguous()
        return unpatched_fn(self)

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

    # work around lack of jit support for torch._C._infer_size()
    # this can be deleted after https://github.com/pytorch/pytorch/pull/10321
    @_patch('torch.distributions.categorical.Categorical.log_prob')
    def _log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    # work around lack of jit support for torch._C._infer_size()
    # this can be deleted after https://github.com/pytorch/pytorch/pull/10321
    @_patch('torch.distributions.multivariate_normal.MultivariateNormal.__init__')
    def _MultivariateNormal_init(self, loc, covariance_matrix=None, precision_matrix=None,
                                 scale_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")

        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self._unbroadcasted_scale_tril = scale_tril
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self._unbroadcasted_scale_tril = torch.distributions.multivariate_normal._batch_potrf_lower(
                covariance_matrix)
            self.covariance_matrix, loc_ = torch.broadcast_tensors(covariance_matrix, loc_)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            covariance_matrix = torch.distributions.multivariate_normal._batch_inverse(precision_matrix)
            self._unbroadcasted_scale_tril = torch.distributions.multivariate_normal._batch_potrf_lower(
                covariance_matrix)
            self.covariance_matrix, self.precision_matrix, loc_ = torch.broadcast_tensors(
                covariance_matrix, precision_matrix, loc_)
        self.loc = loc_[..., 0]  # drop rightmost dim

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(torch.distributions.multivariate_normal.MultivariateNormal, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    # work around lack of jit support for torch._C._infer_size()
    # this can be deleted after https://github.com/pytorch/pytorch/pull/10321
    @_patch('torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.__init__')
    def __init__(self, loc, cov_factor, cov_diag, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:]
        if cov_factor.dim() < 2:
            raise ValueError("cov_factor must be at least two-dimensional, "
                             "with optional leading batch dimensions")
        if cov_factor.shape[-2:-1] != event_shape:
            raise ValueError("cov_factor must be a batch of matrices with shape {} x m"
                             .format(event_shape[0]))
        if cov_diag.shape[-1:] != event_shape:
            raise ValueError("cov_diag must be a batch of vectors with shape {}".format(event_shape))

        loc_ = loc.unsqueeze(-1)
        cov_diag_ = cov_diag.unsqueeze(-1)
        try:
            loc_, self.cov_factor, cov_diag_ = torch.broadcast_tensors(loc_, cov_factor, cov_diag_)
        except RuntimeError:
            raise ValueError("Incompatible batch shapes: loc {}, cov_factor {}, cov_diag {}"
                             .format(loc.shape, cov_factor.shape, cov_diag.shape))
        self.loc = loc_[..., 0]
        self.cov_diag = cov_diag_[..., 0]
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]

        self._capacitance_tril = torch.distributions.lowrank_multivariate_normal._batch_capacitance_tril(
            self.cov_factor, self.cov_diag)
        super(torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)


__all__ = []
