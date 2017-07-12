import torch


class AbstractInfer(object):
    """
    Infer class must implement: _pyro_sample, _pyro_observe,
    _pyro_on_exit, _pyro_param, _pyro_map_data
    """
    def __init__(self):
        pass


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
