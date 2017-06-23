from pyro.infer.poutine import Poutine
import torch

# overload call function here?


class AbstractInfer(object):
    """
    Infer class must implement: _pyro_sample, _pyro_observe,
    _pyro_on_exit, _pyro_param, _pyro_map_data
    """

    def __init__(self):
        pass

# extend Poutine -- switch to this context when appropriate


class LWCopoutine(Poutine):
    # every time
    def _enter_poutine(self, *args, **kwargs):
        """
        When model execution begins
        """
        self.current_score = 0

    def _pyro_sample(self, name, dist):
        """
        Simply sample from distribution
        """
        return dist()

    def _pyro_observe(self, name, dist, val):
        """
        Get log_pdf of sample, add to ongoing scoring
        """
        logp = dist.log_pdf(val)
        self.current_score += logp
        return val


class LikelihoodWeighting(AbstractInfer):

    def __init__(self, model, *args, **kwargs):
        """
        Call parent class initially, then setup the couroutines to run
        """
        # initialize
        super(LikelihoodWeighting, self).__init__()

        # wrap the model function with a LWCoupoutine
        # this will push and pop state
        self.model = LWCopoutine(model)

        # defining here, but will be updated at runner time
        self.current_score = 0.
        self.samples = []

    def runner(self, num_samples, *args, **kwargs):
        """
        Main function of an Infer object, automatically switches context with copoutine
        """
        # setup sample to hold
        samples = []

        for i in range(num_samples):
            # push and pop stack handled by copoutine
            # What about models which take inputs?
            rv = self.model(*args, **kwargs)

            # add to sample state
            samples.append([i, rv, self.model.current_score])

        # send back array of samples to be consumed elsewhere
        return samples

#
# class LWMarginal(object):
#   # takes a trace distribution and consumes
#   def __call__(trace_dist, num_samples, *args, **kwargs):
#     # grab num samples from the trace dist, then consume
# 	  concrete = trace_dist.runner(num_samples)


def lw_expectation(trace_dist, functional, num_samples):
    # running var
    accum_so_far = 0.
    sum_weight = 0.

    # sample from trace_dist
    samples = trace_dist.runner(num_samples)

    # loop over the sample tuples
    for i, rv, cur_score in samples:

        # sum_prob = torch.exp(torch.sum(cur_score))
        sum_prob = torch.exp(cur_score)
        sum_weight += sum_prob

        # apply function to return value, multiply by exp(cur_score)
        accum_so_far += functional(rv) * sum_prob  # .expand_as(functional(rv))

    #
    return accum_so_far / sum_weight  # .expand_as(accum_so_far)
