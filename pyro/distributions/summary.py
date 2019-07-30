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
        Add observed data to this summary. The last batch dimension indicates 
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
        prior_alpha += torch.tensor([0.]) # TODO: hack to handle scalar and tensor inputs
        prior_beta += torch.tensor([0.])
        assert torch.all(prior_alpha > 0.0)
        assert torch.all(prior_beta > 0.0)
        assert prior_alpha.shape == prior_beta.shape
        self.alpha = prior_alpha
        self.beta = prior_beta

    def update(self, obs, features=None):
        assert features is None
        assert obs.shape[-1] == 1
        total = obs.sum([-2,-1])
        self.alpha += total
        self.beta += torch.ones(obs.shape).sum([-2,-1]) - total


class NIGNormalRegressionSummary(Summary):
    """
    Summary of NIG-Normal conjugate family regression data. The prior can be broadcasted to a batch of summaries.

    :param torch.tensor prior_mean: The prior mean parameter for the NIG distribution. 
                                    batch_shape == (other_batches, obs_dim or 1); event_shape == (features.dim)
    :param torch.tensor prior_covariance: The prior covariance parameter for the NIG distribution. 
                                          batch_shape == (other_batches, obs_dim or 1); event_shape == (features.dim, features.dim)
    :param float prior_shape: The prior shape parameter for the NIG distribution. 
                              batch_shape == (other_batches, obs_dim or 1); event_shape is ()
    :param float prior_rate: The prior rate parameter for the NIG distribution. 
                             batch_shape == (other_batches, obs_dim or 1); event_shape is ()
    """
    # TODO: Allow for fast Cholesky rank-1 update
    def __init__(self, prior_mean, prior_covariance, prior_shape, prior_rate):
        # TODO: Hack to allow scalar inputs
        prior_mean += torch.Tensor([0.])
        prior_covariance = prior_covariance + torch.Tensor([[0.]])
        prior_shape += torch.Tensor([0.])
        prior_rate += torch.Tensor([0.])
        assert torch.all(prior_covariance.eq(prior_covariance.transpose(-2, -1)))
        try:
            torch.cholesky(prior_covariance)
        except:
            raise RuntimeError("Covariance is not PSD.")
        
        assert torch.all(prior_shape > 0)
        assert torch.all(prior_rate > 0)
        params = self.convert_to_reparametrized_form(prior_mean, prior_covariance, prior_shape, prior_rate)
        self.precision_times_mean, self.precision, self.shape, self.reparametrized_rate = params

    def update(self, obs, features=None):
        # features batch_shape == (other_batches); event_shape == (update_batch, features_dim)
        # obs:     batch_shape == (other_batches); event_shape == (update_batch, obs_dim)
        assert features is not None
        assert obs.shape[:-1] == features.shape[:-1]
        self.precision_times_mean = self.precision_times_mean + obs.transpose(-2,-1).matmul(features) 
        self.precision = self.precision + (features.transpose(-2,-1).matmul(features)).unsqueeze(-3)
        self.shape = self.shape + 0.5 * obs.shape[-2]
        self.reparametrized_rate = self.reparametrized_rate + 0.5 * (obs * obs).sum(-2)

    @staticmethod
    def convert_to_reparametrized_form(mean, covariance, shape, rate):
        """
        Converts the NIG parameters to a more convenient form which obviates the
        need for matrix inverses needed for updates.

        :returns: a reparametrization of the parameters for faster updating and sampling.
        :rtype: a tuple of precision_times_mean, precision, shape, and reparametrized_rate.
        """
        precision = covariance.inverse() # event_shape == (features.dim, features.dim)
        precision_times_mean = precision.matmul(mean.unsqueeze(-1)).squeeze(-1) # event_shape == (features.dim)
        reparametrized_rate = rate + 0.5*(mean.unsqueeze(-2)).matmul(precision).matmul(mean.unsqueeze(-1)).squeeze(-1).squeeze(-1) # event_shape == ()

        return precision_times_mean, precision, shape, reparametrized_rate

    @staticmethod
    def convert_to_canonical_form(prec_times_mean, precision, shape, reparametrized_rate):
        """
        Converts the NIG parameters back to its canonical form.

        :returns: the canonical parameters.
        :rtype: a tuple of mean, covariance, shape, and rate.
        """
        covariance = precision.inverse()
        mean = covariance.matmul(prec_times_mean.unsqueeze(-1)).squeeze(-1)
        rate = reparametrized_rate - 0.5*(mean.unsqueeze(-2)).matmul(precision).matmul(mean.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        return mean, covariance, shape, rate
