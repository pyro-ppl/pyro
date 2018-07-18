from __future__ import absolute_import, division, print_function

from .messenger import Messenger


class ScaleMessenger(Messenger):
    """
    Given a stochastic function with some sample statements and a positive
    scale factor, scale the score of all sample and observe sites in the
    function.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s), obs=1.0)
        ...     return z ** 2

    ``scale`` multiplicatively scales the log-probabilities of sample sites:

        >>> scaled_model = scale(model, scale=0.5)
        >>> scaled_tr = trace(scaled_model).get_trace(0.0)
        >>> unscaled_tr = trace(model).get_trace(0.0)
        >>> bool((scaled_tr.log_prob_sum() == 0.5 * unscaled_tr.log_prob_sum()).all())
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param scale: a positive scaling factor
    :returns: stochastic function decorated with a :class:`~pyro.poutine.scale_messenger.ScaleMessenger`
    """
    def __init__(self, scale):
        super(ScaleMessenger, self).__init__()
        self.scale = scale

    def _process_message(self, msg):
        msg["scale"] = self.scale * msg["scale"]
        return None
