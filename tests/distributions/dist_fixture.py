import math
import numbers

import numpy as np
import torch
from torch.autograd import Variable


class Fixture(object):
    def __init__(self,
                 pyro_dist=None,
                 scipy_dist=None,
                 dist_params=None,
                 test_data=None,
                 scipy_arg_fn=None,
                 prec=0.05,
                 min_samples=None,
                 is_discrete=False,
                 expected_support_non_vec=None,
                 expected_support=None):
        self.pyro_dist = pyro_dist
        self.scipy_dist = scipy_dist
        self.dist_params = dist_params
        self.test_data = test_data
        self.scipy_arg_fn = scipy_arg_fn
        self.min_samples = min_samples
        self.prec = prec
        self.is_discrete = is_discrete
        self.expected_support_non_vec = expected_support_non_vec
        self.expected_support = expected_support

    def get_samples(self, num_samples, *dist_params):
        return [self.pyro_dist(*dist_params).data.cpu().numpy() for _ in range(num_samples)]

    def get_test_data(self, idx=None):
        test_data = self.test_data
        if idx is not None:
            test_data = self.test_data[idx]
        return tensor_wrap(test_data)

    def get_dist_params(self, idx=None):
        dist_params = self.dist_params
        if idx is not None:
            return map_tensor_wrap(dist_params[idx])
        return map_tensor_wrap(zip(*dist_params))

    def get_scipy_logpdf(self, idx):
        if not self.scipy_arg_fn:
            return
        args, kwargs = self.scipy_arg_fn(*self.dist_params[idx])
        if self.is_discrete:
            log_pdf = self.scipy_dist.logpmf(self.test_data[idx], *args, **kwargs)
        else:
            log_pdf = self.scipy_dist.logpdf(self.test_data[idx], *args, **kwargs)
        return np.sum(log_pdf)

    def get_scipy_batch_logpdf(self):
        if not self.scipy_arg_fn:
            return
        return [self.get_scipy_logpdf(i) for i in range(len(self.test_data))]

    def get_pyro_logpdf(self, idx):
        return self.pyro_dist.log_pdf(self.get_test_data(idx), *self.get_dist_params(idx))

    def get_pyro_batch_logpdf(self):
        return self.pyro_dist.batch_log_pdf(self.get_test_data(), *self.get_dist_params())

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
        args, kwargs = self.scipy_arg_fn(*self.dist_params[idx])
        try:
            fourth_moment = np.max(self.scipy_dist.moment(4, *args, **kwargs))
            var = np.max(self.scipy_dist.var(*args, **kwargs))
            min_computed_samples = int(math.ceil((fourth_moment - math.pow(var, 2)) / required_precision))
        except (AttributeError, ValueError):
            return min_samples
        return max(min_samples, min_computed_samples)

    def get_test_distribution_name(self):
        return self.pyro_dist.__class__.__name__


def tensor_wrap(value):
    if isinstance(value, numbers.Number):
        return Variable(torch.Tensor([value]))
    return Variable(torch.Tensor(value))


def map_tensor_wrap(list_vals):
    return [tensor_wrap(x) for x in list_vals]
