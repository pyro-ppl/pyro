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
        Add data to this summary.
        :param torch.Tensor data: The dimensions are batch_dim x obs_dim
        :param torch.Tensor features: The dimensions are batch_dim x features_dim
        """
        pass


class BetaBernoulliSummary(Summary):
    """
    Summary of Beta-Bernoulli conjugate family data.
    """
    def __init__(self, prior_alpha, prior_beta):
        assert prior_alpha > 0.0
        assert prior_beta > 0.0
        self.alpha = prior_alpha
        self.beta = prior_beta

    def update(self, data, features=None):
        assert features is None
        assert data.dim() == 2
        assert data.shape[1] == 1
        total = data.sum()
        self.alpha += total
        self.beta += data.shape[0] - total


class NIGNormalRegressionSummary(Summary):
    """
    Summary of NIG-Normal conjugate family regression data.
    """
    # TODO: Allow for fast Cholesky rank-1 update
    def __init__(self, prior_mean, prior_covariance, prior_shape, prior_rate):
        # TODO: Should I modify these assertions to allow for scalar inputs?
        assert torch.all(prior_covariance.eq(prior_covariance.t()))
        eig = torch.eig(prior_covariance).eigenvalues
        assert torch.all(eig[:, 0] >= 0.0)
        assert torch.all(eig[:, 1] == 0.0)
        assert prior_shape > 0
        assert prior_rate > 0
        params = self.convert_to_reparameterized_form(prior_mean, prior_covariance, prior_shape, prior_rate)
        (self.precision_times_mean, self.precision, self.shape, self.reparameterized_rate) = params

    def update(self, data, features=None):
        assert features is not None
        assert data.dim() == 2
        assert data.shape[1] == 1
        assert features.dim() == 2
        assert data.shape[0] == features.shape[0]
        self.precision_times_mean += features.t().matmul(data.flatten())
        self.precision += features.t().matmul(features)
        self.shape += 0.5 * data.shape[0]
        self.reparameterized_rate += 0.5 * data.flatten().dot(data.flatten())

    @staticmethod
    def convert_to_reparameterized_form(mean, covariance, shape, rate):
        """
        Converts the NIG parameters to a more convenient form which obviates the
        need for matrix inverses needed for updates.
        """
        precision = covariance.inverse()
        precision_times_mean = precision.matmul(mean)
        reparameterized_rate = rate + 0.5*mean.matmul(precision).matmul(mean)

        return precision_times_mean, precision, shape, reparameterized_rate

    @staticmethod
    def convert_to_canonical_form(prec_times_mean, precision, shape, reparameterized_rate):
        """
        Converts the NIG parameters back to its canonical form.
        """
        covariance = precision.inverse()
        mean = covariance.matmul(prec_times_mean)
        rate = reparameterized_rate - 0.5 * mean.matmul(precision).matmul(mean)

        return mean, covariance, shape, rate
