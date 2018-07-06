import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance


def SearchEIG(model, d):
    # TODO handle pyro parameter right
    # design = pyro.param("design", d)
    def entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        conditioned_model = pyro.condition(model, data={"y": y})
        posterior = Search(conditioned_model).run(design)
        latent_posterior_dist = EmpiricalMarginal(posterior, 
                                                  sites="fairness")
        return integer_rv_entropy(latent_posterior_dist)

    y_dist = EmpiricalMarginal(Search(model).run(d))
    # Why do you need to do this?
    # EmpiricalMarginal makes the sample space appear larger than it is
    # TODO: What about the case where Y is not binary?
    p = torch.exp(y_dist.log_prob(1.))
    y_dist = dist.Bernoulli(p)
    # Want the entropy distribution when y follows its
    # empirical
    loss_dist = EmpiricalMarginal(Search(entropy).run(y_dist, d))
    # Now take the expectation
    loss = 0.
    for v in loss_dist.enumerate_support():  # could sample instead
        loss += v * torch.exp(loss_dist.log_prob(v))
    base_entropy = integer_rv_entropy(EmpiricalMarginal(
                                      Search(model).run(d), 
                                      sites='fairness'))
    return base_entropy - loss


def integer_rv_entropy(dist):
    """Computes the entropy of a random variable supported on Z.
    """
    e = 0.
    supp = set(int(x) for x in dist.enumerate_support())
    # print([(v, torch.exp(dist.log_prob(v))) for v in supp])
    for v in supp:
        lp = dist.log_prob(v)
        if not lp.eq(float('-inf')).any() and not torch.isnan(lp).any():
            e += -lp * torch.exp(lp)
    return e