import torch
from pyro.distributions import constraints
from pyro.distributions.torch import Categorical


class OrderedLogistic(Categorical):
    """
    Alternative parametrization of the distribution over a categorical variable.

    Instead of the typical parametrization of a categorical variable in terms
    of the probability mass of the individual categories ``p``, this provides an
    alternative that is useful in specifying ordered categorical models. This
    accepts a vector of ``cutpoints`` which are an ordered vector of real numbers
    denoting baseline cumulative log-odds of the individual categories, and a
    model vector ``predictors`` which modifies the baselines for each sample
    individually.

    These cumulative log-odds are then transformed into a discrete cumulative
    probability distribution, that is finally differenced to return the probability
    mass matrix ``p`` that specifies the categorical distribution.

    :param Tensor predictors: A tensor of predictor variables of arbitrary
        shape. The output shape of non-batched samples from this distribution will
        be the same shape as ``predictors``.
    :param Tensor cutpoints: A tensor of cutpoints that are used to determine the
        cumulative probability of each entry in ``predictors`` belonging to a
        given category. Must be 1-dimensional and monotonically increasing.
    """

    arg_constraints = {
        "predictors": constraints.real,
        "cutpoints": constraints.ordered_vector,
    }

    def __init__(self, predictors, cutpoints, validate_args=None):
        self.predictors = predictors
        self.cutpoints = cutpoints
        p_shape = predictors.shape + (len(cutpoints) + 1,)
        cutpoints = cutpoints.reshape(predictors.ndim * (1,) + (-1,))
        predictors = predictors.reshape(predictors.shape + (1,))
        # calculate cumulative probability for each sample
        q = torch.sigmoid(cutpoints - predictors)
        # turn cumulative probabilities into probability mass of categories
        p = torch.zeros(p_shape)
        p[..., 0] = q[..., 0]
        p[..., 1:-1] = (q - torch.roll(q, 1, dims=-1))[..., 1:]
        p[..., -1] = 1 - q[..., -1]
        super(OrderedLogistic, self).__init__(p, validate_args=validate_args)
