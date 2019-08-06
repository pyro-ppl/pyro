from abc import ABCMeta, abstractmethod

import torch
from six import add_metaclass
from torch.testing import assert_allclose
from pyro.distributions.util import broadcast_shape, validation_enabled


@add_metaclass(ABCMeta)
class Summary(object):
    """
    Abstract base class for collated sufficient statistics of data.
    """

    @abstractmethod
    def update(self, obs, features=None):
        """
        Add observed data to this summary. The last batch dimension indicates
        a batch update of datapoints.

        :param torch.Tensor obs: The dimensions are batch_dim x obs_dim
        :param torch.Tensor features: The dimensions are batch_dim x features_dim
        """
        pass

    @abstractmethod
    def downdate(self, obs, features=None):
        """
        Remove observed data to this summary. The last batch dimension indicates
        a batch update of datapoints.

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
        self._alpha = prior_alpha + torch.tensor([0.])   # hack to handle scalar and tensor inputs
        self._beta = prior_beta + torch.tensor([0.])
        assert torch.all(self._alpha > 0.0)
        assert torch.all(self._beta > 0.0)

        self._alpha = prior_alpha
        self._beta = prior_beta

    def update(self, obs, features=None):
        assert features is None
        assert obs.shape[-1] == 1
        total = obs.sum([-2, -1])
        self._alpha += total
        self._beta += torch.ones(obs.shape).sum([-2, -1]) - total

    def downdate(self, obs, features=None):
        assert features is None
        assert obs.shape[-1] == 1
        total = obs.sum([-2, -1])
        self._alpha -= total
        self._beta -= torch.ones(obs.shape).sum([-2, -1]) - total

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta


class NIGNormalRegressionSummary(Summary):
    """
    Summary of NIG-Normal conjugate family regression data. The prior can be broadcasted to a batch of summaries.

    :param torch.tensor prior_mean: The prior mean parameter for the NIG distribution.
                                    batch_shape == (other_batches, obs_dim or 1); event_shape == (features.dim)
    :param torch.tensor prior_covariance: The prior covariance parameter for the NIG distribution.
                                          batch_shape == (other_batches, obs_dim or 1);
                                          event_shape == (features.dim, features.dim)
    :param float prior_shape: The prior shape parameter for the NIG distribution.
                              batch_shape == (other_batches, obs_dim or 1); event_shape is ()
    :param float prior_rate: The prior rate parameter for the NIG distribution.
                             batch_shape == (other_batches, obs_dim or 1); event_shape is ()
    """
    # TODO: Allow for fast Cholesky rank-1 update
    def __init__(self, prior_mean, prior_covariance, prior_shape, prior_rate):
        # Hack to allow scalar inputs
        self._mean = torch.as_tensor(prior_mean)
        self._covariance = torch.as_tensor(prior_covariance)
        self._shape = torch.as_tensor(prior_shape)
        self._rate = torch.as_tensor(prior_rate)
        if validation_enabled():
            assert torch.all(self._covariance.eq(self._covariance.transpose(-2, -1)))
            assert torch.all(self._shape > 0)
            assert torch.all(self._rate > 0)

        # Reparametrize
        self._precision = self._covariance.inverse()
        self._precision_times_mean = self._precision.matmul(self._mean.unsqueeze(-1)).squeeze(-1)
        self._reparametrized_rate = self._rate + (0.5*(self._mean.unsqueeze(-2)).matmul(self._precision)
                                                  .matmul(self._mean.unsqueeze(-1)).squeeze(-1).squeeze(-1))

        self._updated_canonical = True
        self.obs_dim = None

    def update(self, obs, features=None):
        # features batch_shape == (other_batches); event_shape == (update_batch, features_dim)
        # obs:     batch_shape == (other_batches); event_shape == (update_batch, obs_dim)
        assert features is not None
        assert obs.dim() >= 2
        assert features.dim() >= 2
        assert obs.size(-2) == features.size(-2)
        if self.obs_dim is None:
            self.obs_dim = obs.shape[-1]
        else:
            assert self.obs_dim == obs.size(-1)

        # Hack to enforce that features and obs are all the same batch dimension such that the parameters
        # are all the same dimension
        batch_shape = broadcast_shape(features.shape[:-1], obs.shape[:-1])
        if features.shape[:-1] != batch_shape:
            features = features.expand(batch_shape + (-1,))
        if obs.shape[:-1] != batch_shape:
            obs = obs.expand(batch_shape + (-1,))

        self._precision_times_mean = self._precision_times_mean + obs.transpose(-2, -1).matmul(features)
        self._precision = self._precision + (features.transpose(-2, -1).matmul(features)).unsqueeze(-3)
        self._shape = self._shape + 0.5 * obs.shape[-2]
        self._reparametrized_rate = self._reparametrized_rate + 0.5 * (obs * obs).sum(-2)

        self._updated_canonical = False

    def downdate(self, obs, features=None):
        # features batch_shape == (other_batches); event_shape == (update_batch, features_dim)
        # obs:     batch_shape == (other_batches); event_shape == (update_batch, obs_dim)
        assert features is not None
        assert obs.dim() >= 2
        assert features.dim() >= 2
        assert obs.shape[-2] == features.shape[-2]
        if self.obs_dim is None:
            self.obs_dim = obs.shape[-1]
        else:
            assert self.obs_dim == obs.shape[-1]

        # Hack to enforce that features and obs are all the same batch dimension such that the parameters
        # are all the same dimension
        features = features + torch.zeros(list(obs.shape[:-1]) + [features.shape[-1]])
        obs = obs + torch.zeros(list(features.shape[:-1]) + [obs.shape[-1]])

        self._precision_times_mean = self._precision_times_mean - obs.transpose(-2, -1).matmul(features)
        self._precision = self._precision - (features.transpose(-2, -1).matmul(features)).unsqueeze(-3)
        self._shape = self._shape - 0.5 * obs.shape[-2]
        self._reparametrized_rate = self._reparametrized_rate - 0.5 * (obs * obs).sum(-2)

        self._updated_canonical = False

    @property
    def mean(self):
        if not self._updated_canonical:
            self._convert_to_canonical_form()
        return self._mean

    @property
    def covariance(self):
        if not self._updated_canonical:
            self._convert_to_canonical_form()
        return self._covariance

    @property
    def rate(self):
        if not self._updated_canonical:
            self._convert_to_canonical_form()
        return self._rate

    @property
    def precision_times_mean(self):
        return self._precision_times_mean

    @property
    def precision(self):
        return self._precision

    @property
    def shape(self):
        return self._shape

    @property
    def reparametrized_rate(self):
        return self._reparametrized_rate

    def _convert_to_canonical_form(self):
        """
        Converts the NIG parameters back to its canonical form.

        :returns: the canonical parameters.
        :rtype: a tuple of mean (features.dim), covariance (features.dim, features.dim),
                shape (), and rate ().
        """
        self._covariance = self._precision.inverse()
        self._mean = self._covariance.matmul(self._precision_times_mean.unsqueeze(-1)).squeeze(-1)
        self._rate = self._reparametrized_rate - (0.5*(self._mean.unsqueeze(-2)).matmul(self._precision)
                                                  .matmul(self._mean.unsqueeze(-1)).squeeze(-1).squeeze(-1))

        self._updated_canonical = True

        return self._mean, self._covariance, self._shape, self._rate

    def __getitem__(self, index):
        # Maybe add .contiguous()?
        mean = self.mean[index]
        covariance = self.covariance[index]
        shape = self.shape[index]
        rate = self.rate[index]
        return NIGNormalRegressionSummary(self, mean, covariance, shape, rate)
