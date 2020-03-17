# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch
from torch.distributions.utils import logits_to_probs

from pyro.distributions.util import broadcast_shape

SINGLE_TEST_DATUM_IDX = [0]
BATCH_TEST_DATA_IDX = [-1]


class Fixture:
    def __init__(self,
                 pyro_dist=None,
                 scipy_dist=None,
                 examples=None,
                 scipy_arg_fn=None,
                 prec=0.05,
                 min_samples=None,
                 is_discrete=False,
                 expected_support_non_vec=None,
                 expected_support=None,
                 test_data_indices=None,
                 batch_data_indices=None):
        self.pyro_dist = pyro_dist
        self.scipy_dist = scipy_dist
        self.dist_params, self.test_data = self._extract_fixture_data(examples)
        self.scipy_arg_fn = scipy_arg_fn
        self.min_samples = min_samples
        self.prec = prec
        self.is_discrete = is_discrete
        self.expected_support_non_vec = expected_support_non_vec
        self.expected_support = expected_support
        self.test_data_indices = test_data_indices
        self.batch_data_indices = batch_data_indices

    def get_batch_data_indices(self):
        if not self.batch_data_indices:
            return BATCH_TEST_DATA_IDX
        return self.batch_data_indices

    def get_test_data_indices(self):
        if not self.test_data_indices:
            return SINGLE_TEST_DATUM_IDX
        return self.test_data_indices

    def _extract_fixture_data(self, examples):
        dist_params, test_data = [], []
        for ex in examples:
            test_data.append(ex.pop('test_data'))
            dist_params.append(ex)
        return dist_params, test_data

    def get_num_test_data(self):
        return len(self.test_data)

    def get_samples(self, num_samples, **dist_params):
        return self.pyro_dist(**dist_params).sample(sample_shape=torch.Size((num_samples,)))

    def get_test_data(self, idx, wrap_tensor=True):
        if not wrap_tensor:
            return self.test_data[idx]
        return tensor_wrap(self.test_data[idx])[0]

    def get_dist_params(self, idx, wrap_tensor=True):
        if not wrap_tensor:
            return self.dist_params[idx]
        return tensor_wrap(**self.dist_params[idx])

    def _convert_logits_to_ps(self, dist_params):
        if 'logits' in dist_params:
            logits = torch.tensor(dist_params.pop('logits'))
            is_multidimensional = self.get_test_distribution_name() not in ['Bernoulli', 'Geometric']
            probs = logits_to_probs(logits, is_binary=not is_multidimensional)
            dist_params['probs'] = list(probs.detach().cpu().numpy())
        return dist_params

    def get_scipy_logpdf(self, idx):
        if not self.scipy_arg_fn:
            return
        dist_params = self.get_dist_params(idx, wrap_tensor=False)
        dist_params = self._convert_logits_to_ps(dist_params)
        args, kwargs = self.scipy_arg_fn(**dist_params)
        if self.is_discrete:
            log_prob = self.scipy_dist.logpmf(self.get_test_data(idx, wrap_tensor=False), *args, **kwargs)
        else:
            log_prob = self.scipy_dist.logpdf(self.get_test_data(idx, wrap_tensor=False), *args, **kwargs)
        return np.sum(log_prob)

    def get_scipy_batch_logpdf(self, idx):
        if not self.scipy_arg_fn:
            return
        dist_params = self.get_dist_params(idx, wrap_tensor=False)
        dist_params_wrapped = self.get_dist_params(idx)
        dist_params = self._convert_logits_to_ps(dist_params)
        test_data = self.get_test_data(idx, wrap_tensor=False)
        test_data_wrapped = self.get_test_data(idx)
        shape = broadcast_shape(self.pyro_dist(**dist_params_wrapped).shape(), test_data_wrapped.size())
        log_prob = []
        for i in range(len(test_data)):
            batch_params = {}
            for k in dist_params:
                param = np.broadcast_to(dist_params[k], shape)
                batch_params[k] = param[i]
            args, kwargs = self.scipy_arg_fn(**batch_params)
            if self.is_discrete:
                log_prob.append(self.scipy_dist.logpmf(test_data[i], *args, **kwargs))
            else:
                log_prob.append(self.scipy_dist.logpdf(test_data[i], *args, **kwargs))
        return log_prob

    def get_num_samples(self, idx):
        r"""
        Number of samples needed to estimate the population variance within the tolerance limit
        Sample variance is normally distributed http://stats.stackexchange.com/a/105338/71884
        (see warning below).
        Var(s^2) /approx 1/n * (\loc_4 - \scale^4)
        Adjust n as per the tolerance needed to estimate the sample variance
        warning: does not work for some distributions like bernoulli - https://stats.stackexchange.com/a/104911
        use the min_samples for explicitly controlling the number of samples to be drawn
        """
        if self.min_samples:
            return self.min_samples
        min_samples = 1000
        tol = 10.0
        required_precision = self.prec / tol
        if not self.scipy_dist:
            return min_samples
        args, kwargs = self.scipy_arg_fn(**self.get_dist_params(idx, wrap_tensor=False))
        try:
            fourth_moment = np.max(self.scipy_dist.moment(4, *args, **kwargs))
            var = np.max(self.scipy_dist.var(*args, **kwargs))
            min_computed_samples = int(math.ceil((fourth_moment - math.pow(var, 2)) / required_precision))
        except (AttributeError, ValueError):
            return min_samples
        return max(min_samples, min_computed_samples)

    def get_test_distribution_name(self):
        return self.pyro_dist.__name__


def tensor_wrap(*args, **kwargs):
    tensor_list, tensor_map = [], {}
    for arg in args:
        wrapped_arg = torch.tensor(arg) if isinstance(arg, list) else arg
        tensor_list.append(wrapped_arg)
    for k in kwargs:
        kwarg = kwargs[k]
        wrapped_kwarg = torch.tensor(kwarg) if isinstance(kwarg, list) else kwarg
        tensor_map[k] = wrapped_kwarg
    if args and not kwargs:
        return tensor_list
    if kwargs and not args:
        return tensor_map
    return tensor_list, tensor_map
