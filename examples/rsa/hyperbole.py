import torch

import collections

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, Search


######################################
# models
######################################

# hashable state
State = collections.namedtuple("State", ["price", "valence"])


def price_prior():
    values = [50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001]
    probs = torch.tensor([0.4205, 0.3865, 0.0533, 0.0538, 0.0223, 0.0211, 0.0112, 0.0111, 0.0083, 0.0120])
    ix = pyro.sample("price", dist.Categorical(probs=probs))
    return values[ix]


def valence_prior(state):
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
    return pyro.sample("valence", dist.Bernoulli(probs[state])).item() == 1


def meaning(utterance, price):
    return utterance == price


def approx(x, b=None):
    if b is None:
        b = 10.
    div = x/b
    rounded = float(int(div)) + 1 if div - float(int(div)) > 0.5 else div
    return b * rounded


qud_fns = {
    "price": lambda state: State(price=state.price, valence=None),
    "valence": lambda state: State(valence=state.valence, price=None),
    "priceValence": lambda state: State(price=state.price, valence=state.valence),
    "approxPrice": lambda state: State(price=approx(state.price), valence=None),
    "approxPriceValence": lambda state: State(price=approx(state.price), valence=state.valence),
}


def qud_prior():
    values = ["price", "valence", "priceValence", "approxPrice", "approxPriceValence"]
    ix = pyro.sample("qud", dist.Categorical(torch.ones(len(values)) / len(values)))
    return values[ix]


def utterance_cost(numberUtt):
    preciseNumberCost = 1
    return 0 if approx(numberUtt) == numberUtt else preciseNumberCost


def utterance_prior():
    utterances = [50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001]
    utteranceProbs = torch.exp(-torch.tensor(utterances, dtype=torch.float32))
    ix = pyro.sample("utterance", dist.Categorical(probs=utteranceProbs))
    return utterances[ix]


def literal_listener(utterance, qud):
    price = price_prior()
    state = State(price=price, valence=valence_prior(price))
    factor("literal_meaning", 0. if meaning(utterance, state.price) else -9999.)
    return qud_fns[qud](state)


def speaker(qudValue, qud):
    alpha = 1.
    utterance = utterance_prior()
    with poutine.block():
        literal_marginal = HashingMarginal(
            Search(literal_listener).run(utterance, qud))
    with poutine.scale(scale=torch.tensor(alpha)):
        pyro.sample("listener", literal_marginal, obs=qudValue)
    return utterance


def pragmatic_listener(utterance):
    # priors
    price = price_prior()
    valence = valence_prior(price)
    qud = qud_prior()

    # model
    state = State(price=price, valence=valence)
    qudValue = qud_fns[qud](state)
    with poutine.block():
        speaker_marginal = HashingMarginal(Search(speaker).run(qudValue, qud))
    pyro.sample("speaker", speaker_marginal, obs=utterance)
    return state


def main():
    listener_posterior = HashingMarginal(Search(pragmatic_listener).run(10000))
    print(listener_posterior())
    dd, vv = listener_posterior._dist_and_values()
    print(dd.probs)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
