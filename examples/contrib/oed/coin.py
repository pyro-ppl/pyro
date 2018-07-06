import torch
from itertools import product

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance

from pyro.contrib.oed.eig import SearchEIG


###################################
# Coin model from tech report
###################################

# Global variable: the number of times to flip the coin
n=4

def coin_model(design):
    # design = pyro.param('design', torch.zeros(n))
    fairness = pyro.sample('fairness', 
        dist.Categorical(torch.Tensor([1, 1, 1])))
    fairness = ['fair', 'bias', 'markov'][fairness]

    if fairness == 'fair':
        seq = [pyro.sample("seq_{}".format(i),
                           dist.Bernoulli(0.5))
               for i in range(n)]
    elif fairness == 'bias':
        w = pyro.sample('w', dist.Categorical(torch.ones(20)))
        w = torch.linspace(0, 1, 20)[w]
        seq = [pyro.sample("seq_{}".format(i),
                           dist.Bernoulli(w))
               for i in range(n)]
    elif fairness == 'markov':
        initial = 0.5 # could be random variable
        t = pyro.sample('t', dist.Categorical(torch.ones(20)))
        transition = torch.linspace(0, 1, 20)[t]
        if n > 0:
            seq = [pyro.sample("seq_0", 
                               dist.Bernoulli(initial))]
            for i in range(1, n):
                p = (1-seq[i-1])*transition + seq[i-1]*(1-transition)
                seq.append(pyro.sample("seq_{}".format(i),
                                       dist.Bernoulli(p)))
        else:
            seq = []
    seq = torch.Tensor(seq)
    m = (seq == design).all()
    y = pyro.sample("y", dist.Delta(m.type(torch.FloatTensor)))
    return y

def OED(model, optimizer):

    def EIG(d):
        # TODO handle pyro parameter right
        # design = pyro.param("design", d)
        def entropy(y_dist, design):
            # Important that y_dist is sampled *within* the function
            y = pyro.sample("conditioning_y", y_dist)
            conditioned_model = pyro.condition(model, data={"y": y})
            posterior = Search(conditioned_model).run(design)
            latent_posterior_dist = EmpiricalMarginal(posterior, 
                                                      sites="fairness")
            return rv_entropy(latent_posterior_dist)

        y_dist = EmpiricalMarginal(Importance(model, num_samples=100).run(d))
        # Why do you need to do this?
        # EmpiricalMarginal makes the sample space appear larger than it is
        p = torch.exp(y_dist.log_prob(1.))
        y_dist = dist.Bernoulli(p)
        # Want the entropy distribution when y follows its
        # empirical
        loss_dist = EmpiricalMarginal(Search(entropy).run(y_dist, d))
        # Now take the expectation
        loss = 0.
        for v in loss_dist.enumerate_support():  # could sample instead
            loss += v * torch.exp(loss_dist.log_prob(v))
        base_entropy = rv_entropy(EmpiricalMarginal(
                                  Search(model).run(d), 
                                    sites='fairness'))
        return base_entropy - loss

    space = (torch.Tensor(x) for x in product([0., 1.], repeat=n))
    return optimizer(space, EIG)  # not using existing optimizer, let's go with it for now

def total_enum_optimizer(space, function):
    maximum = None
    maximizer = None
    for point in space:
        f = function(point)
        print(point, f)
        if maximum is None or f > maximum:
            maximum = f
            maximizer = point
    return maximizer, maximum

def rv_entropy(dist):
    e = 0.
    supp = set(int(x) for x in dist.enumerate_support())
    # print([(v, torch.exp(dist.log_prob(v))) for v in supp])
    for v in supp:
        lp = dist.log_prob(v)
        if not lp.eq(float('-inf')).any() and not torch.isnan(lp).any():
            e += -lp * torch.exp(lp)
    return e


if __name__ == '__main__':
    space = (torch.Tensor(x) for x in product([0., 1.], repeat=n))
    for point in space:
        print(point, float(SearchEIG(coin_model, point)))


