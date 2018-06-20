"""
Interpreting hyperbole with RSA models of pragmatics.

Taken from: https://gscontras.github.io/probLang/chapters/03-nonliteral.html
"""

import torch

import collections
import argparse

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, memoize, Search

torch.set_default_dtype(torch.float64)  # double precision for numerical stability


def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))


######################################
# models
######################################

# hashable state
State = collections.namedtuple("State", ["price", "valence"])


def approx(x, b=None):
    if b is None:
        b = 10.
    div = float(x)/b
    rounded = int(div) + 1 if div - float(int(div)) >= 0.5 else int(div)
    return int(b) * rounded


def price_prior():
    values = [50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001]
    probs = torch.tensor([0.4205, 0.3865, 0.0533, 0.0538, 0.0223, 0.0211, 0.0112, 0.0111, 0.0083, 0.0120])
    ix = pyro.sample("price", dist.Categorical(probs=probs))
    return values[ix]


def valence_prior(price):
    probs = {
        50: 0.3173,
        51: 0.3173,
        500: 0.7920,
        501: 0.7920,
        1000: 0.8933,
        1001: 0.8933,
        5000: 0.9524,
        5001: 0.9524,
        10000: 0.9864,
        10001: 0.9864
    }
    return pyro.sample("valence", dist.Bernoulli(probs=probs[price])).item() == 1


def meaning(utterance, price):
    return utterance == price


qud_fns = {
    "price": lambda state: State(price=state.price, valence=None),
    "valence": lambda state: State(price=None, valence=state.valence),
    "priceValence": lambda state: State(price=state.price, valence=state.valence),
    "approxPrice": lambda state: State(price=approx(state.price), valence=None),
    "approxPriceValence": lambda state: State(price=approx(state.price), valence=state.valence),
}


def qud_prior():
    values = ["price", "valence", "priceValence", "approxPrice", "approxPriceValence"]
    ix = pyro.sample("qud", dist.Categorical(probs=torch.ones(len(values)) / len(values)))
    return values[ix]


def utterance_cost(numberUtt):
    preciseNumberCost = 1.
    return 0. if approx(numberUtt) == numberUtt else preciseNumberCost


def utterance_prior():
    utterances = [50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001]
    utteranceLogits = -torch.tensor(list(map(utterance_cost, utterances)),
                                    dtype=torch.float64)
    ix = pyro.sample("utterance", dist.Categorical(logits=utteranceLogits))
    return utterances[ix]


@Marginal
def literal_listener(utterance, qud):
    price = price_prior()
    state = State(price=price, valence=valence_prior(price))
    factor("literal_meaning", 0. if meaning(utterance, price) else -999999.)
    return qud_fns[qud](state)


@Marginal
def speaker(qudValue, qud):
    alpha = 1.
    utterance = utterance_prior()
    literal_marginal = literal_listener(utterance, qud)
    with poutine.scale(scale=torch.tensor(alpha)):
        pyro.sample("listener", literal_marginal, obs=qudValue)
    return utterance


@Marginal
def pragmatic_listener(utterance):
    # priors
    price = price_prior()
    valence = valence_prior(price)
    qud = qud_prior()

    # model
    state = State(price=price, valence=valence)
    qudValue = qud_fns[qud](state)
    speaker_marginal = speaker(qudValue, qud)
    pyro.sample("speaker", speaker_marginal, obs=utterance)
    return state


def test_truth():
    true_vals = {
        "probs": torch.tensor([0.0018655171404222354,0.1512643329444101,0.0030440475496016296,0.23182161303428897,0.00003854830096338984,0.01502495595927897,0.00003889558295405101,0.015160315922876075,0.00016425635615857924,0.026788637869123822,0.00017359794987375924,0.028312162297699582,0.0008164336950199063,0.060558944822420434,0.0008088460212743665,0.05999612935009309,0.01925106279557206,0.17429720083660782,0.02094455861717477,0.18962994295418778]),  # noqa: E231,E501
        "support": list(map(lambda d: State(**d), [{"price":10001,"valence":False},{"price":10001,"valence":True},{"price":10000,"valence":False},{"price":10000,"valence":True},{"price":5001,"valence":False},{"price":5001,"valence":True},{"price":5000,"valence":False},{"price":5000,"valence":True},{"price":1001,"valence":False},{"price":1001,"valence":True},{"price":1000,"valence":False},{"price":1000,"valence":True},{"price":501,"valence":False},{"price":501,"valence":True},{"price":500,"valence":False},{"price":500,"valence":True},{"price":51,"valence":False},{"price":51,"valence":True},{"price":50,"valence":False},{"price":50,"valence":True}]))  # noqa: E231,E501
    }

    pragmatic_marginal = pragmatic_listener(10000)
    for i, elt in enumerate(true_vals["support"]):
        print("{}: true prob {} pyro prob {}".format(
            elt, true_vals["probs"][i].item(),
            pragmatic_marginal.log_prob(elt).exp().item()))


def main(args):

    # test_truth()

    pragmatic_marginal = pragmatic_listener(args.price)

    pd, pv = pragmatic_marginal._dist_and_values()
    print([(s, pragmatic_marginal.log_prob(s).exp().item())
           for s in pragmatic_marginal.enumerate_support()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=10, type=int)
    parser.add_argument('--price', default=10000, type=int)
    args = parser.parse_args()
    main(args)
