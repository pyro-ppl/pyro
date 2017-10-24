import math

import numpy as np
import torch
from torch.autograd import Variable

from tests.common import assert_equal


SINGLE_TEST_DATUM_IDX = [0]
BATCH_TEST_DATA_IDX = [-1]


class Fixture(object):
    def __init__(self,
                 pyro_dist=None,
                 scipy_dist=None,
                 examples=None,
                 scipy_arg_fn=None,
                 prec=0.05,
                 min_samples=None,
                 is_discrete=False,
                 expected_support_non_vec=None,
                 expected_support=None):
        self.pyro_dist, self.pyro_dist_obj = pyro_dist
        self.scipy_dist = scipy_dist
        self.dist_params, self.test_data = self._extract_fixture_data(examples)
        self.scipy_arg_fn = scipy_arg_fn
        self.min_samples = min_samples
        self.prec = prec
        self.is_discrete = is_discrete
        self.expected_support_non_vec = expected_support_non_vec
        self.expected_support = expected_support

    def _extract_fixture_data(self, examples):
        dist_params, test_data = [], []
        for ex in examples:
            test_data.append(ex.pop('test_data'))
            dist_params.append(ex)
        return dist_params, test_data

    def get_num_test_data(self):
        return len(self.test_data)

    def batch_test_data_idx(self):
        return BATCH_TEST_DATA_IDX

    def test_data_idx(self):
        return SINGLE_TEST_DATUM_IDX

    def get_samples(self, num_samples, **dist_params):
        return [self.pyro_dist(**dist_params).data.cpu().numpy() for _ in range(num_samples)]

    def get_test_data(self, idx, wrap_tensor=True):
        if not wrap_tensor:
            return self.test_data[idx]
        return tensor_wrap(self.test_data[idx])[0]

    def get_dist_params(self, idx, wrap_tensor=True):
        if not wrap_tensor:
            return self.dist_params[idx]
        return tensor_wrap(**self.dist_params[idx])

    def get_scipy_logpdf(self, idx):
        if not self.scipy_arg_fn:
            return
        args, kwargs = self.scipy_arg_fn(**self.get_dist_params(idx, wrap_tensor=False))
        if self.is_discrete:
            log_pdf = self.scipy_dist.logpmf(self.get_test_data(idx, wrap_tensor=False), *args, **kwargs)
        else:
            log_pdf = self.scipy_dist.logpdf(self.get_test_data(idx, wrap_tensor=False), *args, **kwargs)
        return np.sum(log_pdf)

    def get_scipy_batch_logpdf(self, idx):
        if not self.scipy_arg_fn:
            return
        dist_params = self.get_dist_params(idx, wrap_tensor=False)
        test_data = self.get_test_data(idx, wrap_tensor=False)
        batch_log_pdf = []
        for i in range(len(test_data)):
            batch_params = {}
            for k in dist_params:
                batch_params[k] = dist_params[k][i]
            args, kwargs = self.scipy_arg_fn(**batch_params)
            if self.is_discrete:
                batch_log_pdf.append(self.scipy_dist.logpmf(test_data[i],
                                                            *args,
                                                            **kwargs))
            else:
                batch_log_pdf.append(self.scipy_dist.logpdf(test_data[i],
                                                            *args,
                                                            **kwargs))
        return batch_log_pdf

    def get_pyro_logpdf(self, idx):
        dist_function_return = self.pyro_dist.log_pdf(self.get_test_data(idx), **self.get_dist_params(idx))
        dist_object_return = self.pyro_dist_obj(**self.get_dist_params(idx)).log_pdf(self.get_test_data(idx))
        assert_equal(dist_function_return, dist_object_return)
        return dist_function_return

    def get_pyro_batch_logpdf(self, idx):
        dist_function_return = self.pyro_dist.batch_log_pdf(self.get_test_data(idx), **self.get_dist_params(idx))
        dist_object_return = self.pyro_dist_obj(**self.get_dist_params(idx)).batch_log_pdf(self.get_test_data(idx))
        assert_equal(dist_function_return, dist_object_return)
        return dist_function_return

    def get_num_samples(self, idx):
        """
        Number of samples needed to estimate the population variance within the tolerance limit
        Sample variance is normally distributed http://stats.stackexchange.com/a/105338/71884
        (see warning below).
        Var(s^2) /approx 1/n * (\mu_4 - \sigma^4)
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
        return self.pyro_dist.__class__.__name__


def tensor_wrap(*args, **kwargs):
    tensor_list, tensor_map = [], {}
    for arg in args:
        if isinstance(arg, list):
            tensor_list.append(Variable(torch.Tensor(arg)))
        else:
            tensor_list.append(arg)
    for k in kwargs:
        if isinstance(kwargs[k], list):
            tensor_map[k] = Variable(torch.Tensor(kwargs[k]))
        else:
            tensor_map[k] = kwargs[k]
    if args and not kwargs:
        return tensor_list
    if kwargs and not args:
        return tensor_map
    return tensor_list, tensor_map
