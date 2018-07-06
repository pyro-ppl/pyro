import torch
import numpy as np

import pyro
import pyro.distributions as dist
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI, Trace_ELBO
from pyro.optim import Adam


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
    # This plays badly with Search
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

softplus = torch.nn.Softplus()

def design_to_matrix(design):
    n, p = int(torch.sum(design)), int(design.size()[0])
    X = torch.zeros(n, p)
    t = 0
    for col, i in enumerate(design):
        i = int(i)
        X[t:t+i, col] = 1.
        t += i
    return X

def ContinuousEIG(model, guide, d, n_steps=100, n_samples=100, vi=True):
    # TODO handle pyro parameter right
    # design = pyro.param("design", d)
    def entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        if vi:
            conditioned_model = pyro.condition(model, data={"y": y})
            optim = Adam({"lr": 0.25})
            posterior = SVI(model, guide, optim, loss=Trace_ELBO())
            with pyro.poutine.block():
                for _ in range(n_steps):
                    posterior.step(design)
            # Recover the entropy
            # TODO: Limited to Normal case atm
            mw_param = pyro.param("guide_mean_weight")
            sw_param = softplus(pyro.param("guide_log_scale_weight"))
            # guide distributions for w 
            # Cannot use the pyro dist :(
            p = int(mw_param.size()[0])

            posterior_cov = sw_param*torch.eye(2)
            w_dist = torch.distributions.MultivariateNormal(mw_param, posterior_cov)
            return w_dist.entropy()
        
        # Compare: compute entropy of posterior analytically
        else:
            prior_cov = torch.Tensor([[1, 0], [0, .1]])
            X = design_to_matrix(design)
            posterior_cov =  prior_cov - prior_cov.mm(X.t().mm(torch.inverse(X.mm(prior_cov.mm(X.t())) + torch.eye(100)).mm(X.mm(prior_cov))))
            print(posterior_cov)
            return 0.5*torch.logdet(2*np.pi*np.e*posterior_cov)
        
    y_dist = EmpiricalMarginal(
        Importance(model, num_samples=n_samples).run(d), sites="y")

    # Want the entropy distribution when y follows its
    # empirical
    loss_dist = EmpiricalMarginal(Search(entropy).run(y_dist, d))
    # Now take the expectation
    loss = 0.
    for v in loss_dist.enumerate_support():  # could sample instead
        loss += v * torch.exp(loss_dist.log_prob(v))
    base_entropy = 0.
    return base_entropy - loss


