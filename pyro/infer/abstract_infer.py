import torch
from torch.autograd import Variable
import pyro
import pyro.util
from pyro.distributions import Categorical


class AbstractInfer(pyro.distributions.Distribution):
    """
    abstract inference class
    TODO documentation
    """
    def _traces(self, *args, **kwargs):
        """
        Virtual method to get unnormalized weighted list of posterior traces
        """
        raise NotImplementedError("inference algorithm must implement _traces")
    
    @pyro.util.memoize
    def _dist(self, *args, **kwargs):
        """
        make trace posterior distribution object with normalized probs
        """
        traces, log_weights = self._traces(*args, **kwargs)
        log_ps = torch.cat(log_weights, 0)
        log_ps = log_ps - pyro.util.log_sum_exp(log_ps).expand_as(log_ps)
        # XXX Categorical not working correctly with non-Tensor vs
        return Categorical(ps=torch.exp(log_ps), vs=[traces])

    def sample(self, *args, **kwargs):
        """
        sample from trace posterior
        """
        return self._dist(*args, **kwargs).sample()

    def log_pdf(self, val, *args, **kwargs):
        """
        Use the histogram to score a value
        """
        return self._dist(*args, **kwargs).log_pdf(val)

    def log_z(self, *args, **kwargs):
        """
        estimate marginal probability of observations
        """
        traces, log_weights = self._traces(*args, **kwargs)
        log_z = 0.0
        # TODO parallelize
        for tr, log_w in zip(traces, log_weights):
            log_z = log_z + log_w
        return log_z / len(traces)


def lw_expectation(trace_dist, functional, num_samples):
    # running var
    accum_so_far = 0.
    sum_weight = 0.

    # sample from trace_dist
    samples = trace_dist.runner(num_samples)

    # loop over the sample tuples
    for i, rv, cur_score in samples:

        # not necessarily efficient torch.exp call x2, fix later
        sum_weight += torch.exp(cur_score)

        # apply function to return value, multiply by exp(cur_score)
        accum_so_far += functional(rv) * torch.exp(cur_score)

    #
    return accum_so_far / sum_weight
