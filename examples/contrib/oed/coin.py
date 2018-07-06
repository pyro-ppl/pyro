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


if __name__ == '__main__':
    space = (torch.Tensor(x) for x in product([0., 1.], repeat=n))
    for point in space:
        print(point, float(SearchEIG(coin_model, point)))


