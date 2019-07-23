from abc import ABCMeta, abstractmethod

import torch
from six import add_metaclass


@add_metaclass(ABCMeta)
class Summary(object):
    """
    Abstract base class for collated sufficient statistics of data.
    """

    @abstractmethod
    def update(self, obs, features=None):
        """
        Add observed data to this summary.
        :param torch.Tensor obs: The dimensions are batch_dim x obs_dim
        :param torch.Tensor features: The dimensions are batch_dim x features_dim
        """
        pass


class BetaBernoulliSummary(Summary):
    """
    Summary of Beta-Bernoulli conjugate family data.

    :param float prior_alpha: The prior alpha parameter for the Beta distribution.
    :param float prior_beta: The prior beta parameter for the Beta distribution.
    """
    def __init__(self, prior_alpha, prior_beta):
        assert prior_alpha > 0.0
        assert prior_beta > 0.0
        self.alpha = prior_alpha
        self.beta = prior_beta

    def update(self, obs, features=None):
        assert features is None
        assert obs.dim() == 2
        assert obs.shape[1] == 1
        total = obs.sum()
        self.alpha += total
        self.beta += obs.shape[0] - total


class NIGNormalRegressionSummary(Summary):
    """
    Summary of NIG-Normal conjugate family regression data.

    :param torch.tensor prior_mean: The prior mean parameter for the NIG distribution.
    :param torch.tensor prior_covariance: The prior covariance parameter for the NIG distribution.
    :param float prior_shape: The prior shape parameter for the NIG distribution.
    :param float prior_rate: The prior rate parameter for the NIG distribution.
    """
    # TODO: Allow for fast Cholesky rank-1 update
    def __init__(self, prior_mean, prior_covariance, prior_shape, prior_rate):
        # TODO: Modify these assertions to allow for scalar inputs?
        assert torch.all(prior_covariance.eq(prior_covariance.t()))
        eig = torch.eig(prior_covariance).eigenvalues
        assert torch.all(eig[:, 0] >= 0.0)
        assert torch.all(eig[:, 1] == 0.0)
        assert prior_shape > 0
        assert prior_rate > 0
        params = self.convert_to_reparameterized_form(prior_mean, prior_covariance, prior_shape, prior_rate)
        (self.precision_times_mean, self.precision, self.shape, self.reparameterized_rate) = params

    def update(self, obs, features=None):
        assert features is not None
        assert obs.dim() == 2
        assert obs.shape[1] == 1
        assert features.dim() == 2
        assert obs.shape[0] == features.shape[0]
        self.precision_times_mean += features.t().matmul(obs.flatten())
        self.precision += features.t().matmul(features)
        self.shape += 0.5 * obs.shape[0]
        self.reparameterized_rate += 0.5 * obs.flatten().dot(obs.flatten())

    @staticmethod
    def convert_to_reparameterized_form(mean, covariance, shape, rate):
        """
        Converts the NIG parameters to a more convenient form which obviates the
        need for matrix inverses needed for updates.

        :returns: a reparameterization of the parameters for faster updating and sampling.
        :rtype: a tuple of precision_times_mean, precision, shape, and reparameterized_rate.
        """
        precision = covariance.inverse()
        precision_times_mean = precision.matmul(mean)
        reparameterized_rate = rate + 0.5*mean.matmul(precision).matmul(mean)

        return precision_times_mean, precision, shape, reparameterized_rate

    @staticmethod
    def convert_to_canonical_form(prec_times_mean, precision, shape, reparameterized_rate):
        """
        Converts the NIG parameters back to its canonical form.

        :returns: the canonical parameters.
        :rtype: a tuple of mean, covariance, shape, and rate.
        """
        covariance = precision.inverse()
        mean = covariance.matmul(prec_times_mean)
        rate = reparameterized_rate - 0.5 * mean.matmul(precision).matmul(mean)

        return mean, covariance, shape, rate
