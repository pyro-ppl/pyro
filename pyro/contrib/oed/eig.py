import torch
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.oed.search import Search
from pyro.infer import EmpiricalMarginal, Importance, SVI, Trace_ELBO
from pyro.optim import Adam


import inspect


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
    loss = loss_dist.mean
    base_entropy = 0.
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
    n, p = int(torch.sum(design)), int(design.shape[0])
    X = torch.zeros(n, p)
    t = 0
    for col, i in enumerate(design):
        i = int(i)
        X[t:t+i, col] = 1.
        t += i
    return X

def ContinuousEIG(model, guide, d, n_steps=3000, n_samples=2, vi=True):
    # TODO handle pyro parameter right
    # design = pyro.param("design", d)
    def entropy(y_dist, design):
        # Important that y_dist is sampled *within* the function
        y = pyro.sample("conditioning_y", y_dist)
        if vi:
            conditioned_model = pyro.condition(model, data={"y": y})
            optim = Adam({"lr": 0.0025})
            posterior = SVI(conditioned_model, guide, optim, loss=Trace_ELBO())
            with pyro.poutine.block():
                for _ in range(n_steps):
                    posterior.step(design)
            # Recover the entropy
            # TODO: Limited to Normal case atm
            mw_param = pyro.param("guide_mean_weight")
            sw_param = softplus(pyro.param("guide_scale_weight"))
            w_dist = dist.Normal(mw_param, sw_param)
            e = w_dist.entropy().sum(-2).sum(-1)
            return e
        
        # Compare: compute entropy of posterior analytically
        # TODO remove temporary code
        else:
            entropies = []
            batch = int(design.shape[0])
            for i in range(batch):
                prior_cov = torch.Tensor([[1, 0], [0, .25]])
                X = design[i,:, :]
                posterior_cov =  prior_cov - prior_cov.mm(X.t().mm(torch.inverse(X.mm(prior_cov.mm(X.t())) + torch.eye(100)).mm(X.mm(prior_cov))))
                entropies.append(0.5*torch.logdet(2*np.pi*np.e*posterior_cov))
            return torch.Tensor(entropies)
        
    y_dist = EmpiricalMarginal(
        Importance(model, num_samples=n_samples).run(d), sites="y")

    # Want the entropy distribution when y follows its
    # empirical
    loss_dist = EmpiricalMarginal(Search(entropy).run(y_dist, d))
    # Now take the expectation
    loss = loss_dist.mean
    prior_cov = torch.Tensor([[1, 0], [0, .25]])
    base_entropy = 0.5*torch.logdet(2*np.pi*np.e*prior_cov)
    return base_entropy - loss


def naiveRainforth(model, design, *args, observation_labels="y", N=100, M=20):

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]

    # 100 traces using batching
    eig = 0.
    for _ in range(N):
        y_given_theta = 0.
        y = {}
        trace = poutine.trace(model).get_trace(design)
        trace.compute_log_prob()
        for label in observation_labels:
            # Valid? Yes, this is log probability conditional on
            # theta, and any previously sampled y's
            # Order doesn't matter
            y_given_theta += trace.nodes[label]["log_prob"]
            y[label] = trace.nodes[label]["value"]

        lp_shape = y_given_theta.shape

        y_given_other_theta = torch.zeros(*lp_shape, M+1)
        y_given_other_theta[..., -1] = y_given_theta
        conditional_model = pyro.condition(model, data=y)
        for j in range(M):
            trace = poutine.trace(conditional_model).get_trace(design)
            trace.compute_log_prob()
            for label in observation_labels:
                y_given_other_theta[..., j] += trace.nodes[label]["log_prob"]

        eig += y_given_theta - torch.distributions.utils.log_sum_exp(
            y_given_other_theta).squeeze() + np.log(M)

    return eig/N











