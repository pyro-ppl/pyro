# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
from tqdm import tqdm

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.ops.indexing import Vindex

smoke_test = ('CI' in os.environ)


def main():
    """
     A toy mixture model to provide a simple example for implementing discrete enumeration.

     (A) -> [B] -> (C)

     A is an observed Bernoulli variable with Beta prior.
     B is a hidden variable which is a mixture of two Bernoulli distributions (with Beta priors),
     chosen by A being true or false.
     C is observed, and like B, is a mixture of two Bernoulli distributions (with Beta priors),
     chosen by B being true or false.
     There is a plate over the three variables for n independent observations of data.

     Because B is hidden and discrete we wish to marginalize it out of the model.
     This is done by:
        1) marking the model method with `@pyro.infer.config_enumerate`
        2) marking the B sample site in the model with `infer={"enumerate": "parallel"}`
        3) passing `pyro.infer.SVI` the `pyro.infer.TraceEnum_ELBO` loss function
    """

    # number of observations
    n = 10000

    prior, CPDs, data = generate_data(n)

    posterior_params = train(prior, data, n)

    evaluate(CPDs, posterior_params)


def generate_data(n):

    # domain = [False, True]
    prior = {'A': torch.tensor([1., 10.]),
             'B': torch.tensor([[10., 1.],
                                [1., 10.]]),
             'C': torch.tensor([[10., 1.],
                                [1., 10.]]),
             }

    # CPDs
    CPDs = {'p_A': Beta(prior['A'][0], prior['A'][1]).sample(),
            'p_B': Beta(prior['B'][:, 0], prior['B'][:, 1]).sample(),
            'p_C': Beta(prior['C'][:, 0], prior['C'][:, 1]).sample(),
            }

    data = {}
    data['A'] = Bernoulli(torch.ones(n) * CPDs['p_A']).sample()
    data['B'] = Bernoulli(torch.gather(CPDs['p_B'], 0, data['A'].type(torch.long))).sample()
    data['C'] = Bernoulli(torch.gather(CPDs['p_C'], 0, data['B'].type(torch.long))).sample()

    return prior, CPDs, data


@pyro.infer.config_enumerate
def model(prior, obs, n):

    p_A = pyro.sample('p_A', dist.Beta(1, 1))

    p_B = pyro.sample('p_B', dist.Beta(torch.ones(2), torch.ones(2)).to_event(1))

    p_C = pyro.sample('p_C', dist.Beta(torch.ones(2), torch.ones(2)).to_event(1))

    with pyro.plate('data_plate', n):
        A = pyro.sample('A', dist.Bernoulli(p_A.expand(n)), obs=obs['A'])

        B = pyro.sample('B', dist.Bernoulli(Vindex(p_B)[A.type(torch.long)]), infer={"enumerate": "parallel"})

        pyro.sample('C', dist.Bernoulli(Vindex(p_C)[B.type(torch.long)]), obs=obs['C'])


def guide(prior, obs, n):

    a = pyro.param('a', prior['A'], constraint=constraints.positive)
    pyro.sample('p_A', dist.Beta(a[0], a[1]))

    b = pyro.param('b', prior['B'], constraint=constraints.positive)
    pyro.sample('p_B', dist.Beta(b[:, 0], b[:, 1]).to_event(1))

    c = pyro.param('c', prior['C'], constraint=constraints.positive)
    pyro.sample('p_C', dist.Beta(c[:, 0], c[:, 1]).to_event(1))


def train(prior, data, n):
    pyro.enable_validation(True)
    pyro.clear_param_store()

    # max_plate_nesting = 1 because there is a single plate in the model
    loss_func = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)

    # setup svi object
    svi = pyro.infer.SVI(model,
                         guide,
                         pyro.optim.Adam({}),
                         loss=loss_func
                         )

    # perform svi
    num_steps = 20000 if not smoke_test else 1
    losses = []
    for _ in tqdm(range(num_steps)):
        loss = svi.step(prior, data, n)
        losses.append(loss)

    plt.figure()
    plt.plot(losses)
    plt.show()

    posterior_params = {k: np.array(v.data) for k, v in pyro.get_param_store().items()}
    posterior_params['a'] = posterior_params['a'][None, :]  # reshape to same as other variables

    return posterior_params


def evaluate(CPDs, posterior_params):

    true_p, pred_p = get_true_pred_CPDs(CPDs['p_A'], posterior_params['a'])
    print('\np_A = True')
    print('actual:   ', true_p)
    print('predicted:', pred_p)

    true_p, pred_p = get_true_pred_CPDs(CPDs['p_B'], posterior_params['b'])
    print('\np_B = True | A = False/True')
    print('actual:   ', true_p)
    print('predicted:', pred_p)

    true_p, pred_p = get_true_pred_CPDs(CPDs['p_C'], posterior_params['c'])
    print('\np_C = True | B = False/True')
    print('actual:   ', true_p)
    print('predicted:', pred_p)


def get_true_pred_CPDs(CPD, posterior_param):
    true_p = CPD.numpy()
    pred_p = posterior_param[:, 0]/np.sum(posterior_param, axis=1)
    return true_p, pred_p


if __name__ == '__main__':
    main()
