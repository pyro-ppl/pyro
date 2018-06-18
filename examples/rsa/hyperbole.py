"""
///fold:
// Round x to nearest multiple of b (used for approximate interpretation):
var approx = function(x,b) {
  var b = 10
  return b * Math.round(x / b)
}

// Here is the code from the Kao et al. hyperbole model
// Prior probability of kettle prices (taken from human experiments)
var pricePrior = function() {
  return categorical({
    vs: [
      50, 51,
      500, 501,
      1000, 1001,
      5000, 5001,
      10000, 10001
    ],
    ps: [
      0.4205, 0.3865,
      0.0533, 0.0538,
      0.0223, 0.0211,
      0.0112, 0.0111,
      0.0083, 0.0120
    ]
  })
}

// Probability that given a price state, the speaker thinks it's too
// expensive (taken from human experiments)
var valencePrior = function(state) {
  var probs = {
    50 : 0.3173,
    51 : 0.3173,
    500 : 0.7920,
    501 : 0.7920,
    1000 : 0.8933,
    1001 : 0.8933,
    5000 : 0.9524,
    5001 : 0.9524,
    10000 : 0.9864,
    10001 : 0.9864
  }
  var tf = flip(probs[state])
  return tf
}

// Literal interpretation "meaning" function;
// checks if uttered number reflects price state
var meaning = function(utterance, price) {
  return utterance == price
}

var qudFns = {
  price : function(state) {return { price: state.price } },
  valence : function(state) {return { valence: state.valence } },
  priceValence : function(state) {
    return { price: state.price, valence: state.valence }
  },
  approxPrice : function(state) {return { price: approx(state.price) } },
  approxPriceValence: function(state) {
    return { price: approx(state.price), valence: state.valence  }
  }
}
///

// Prior over QUDs
var qudPrior = function() {
 categorical({
    vs: ["price", "valence", "priceValence", "approxPrice", "approxPriceValence"],
    ps: [1, 1, 1, 1, 1]
  })
}

// Define list of possible utterances (same as price states)
var utterances = [
  50, 51,
  500, 501,
  1000, 1001,
  5000, 5001,
  10000, 10001
]

// precise numbers can be assumed to be costlier than round numbers
var preciseNumberCost = 1
var utteranceCost = function(numberUtt){
  return numberUtt == approx(numberUtt) ? // if it's a round number utterance
        0 : // no cost
        preciseNumberCost // cost of precise numbers (>= 0)
}

var utteranceProbs = map(function(numberUtt){
  return Math.exp(-utteranceCost(numberUtt)) // prob ~ e^(-cost)
}, utterances)

var utterancePrior = function() {
  categorical({ vs: utterances, ps: utteranceProbs })
}

// Literal listener, infers the qud value assuming the utterance is
// true of the state
var literalListener = cache(function(utterance, qud) {
  return Infer({model: function(){
    var price = pricePrior() // uncertainty about the price
    var valence = valencePrior(price) // uncertainty about the valence
    var fullState = {price, valence}
    condition( meaning(utterance, price) )
    var qudFn = qudFns[qud]
    return qudFn(fullState)
  }
})})

// set speaker optimality
var alpha = 1

// Speaker, chooses an utterance to convey a particular value of the qud
var speaker = cache(function(qudValue, qud) {
  return Infer({model: function(){
    var utterance = utterancePrior()
    factor(alpha*literalListener(utterance,qud).score(qudValue))
    return utterance
  }
})})

// Pragmatic listener, jointly infers the price state, speaker valence, and QUD
var pragmaticListener = cache(function(utterance) {
  return Infer({model: function(){
    //////// priors ////////
    var price = pricePrior()
    var valence = valencePrior(price)
    var qud = qudPrior()
    ////////////////////////
    var fullState = {price, valence}
    var qudFn = qudFns[qud]
    var qudValue = qudFn(fullState)
    observe(speaker(qudValue, qud), utterance)
    return fullState
  }
})})

var listenerPosterior = pragmaticListener(10000)

print("pragmatic listener's joint interpretation of 'The kettle cost $10,000':")
viz(listenerPosterior)
"""
import torch

import collections

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from .search_inference import factor, HashingMarginal, Search


######################################
# models
######################################

# hashable state
State = collections.namedtuple("State", ["price", "valence"])


def price_prior():
    values = torch.tensor([50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001])
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
    return pyro.sample("valence", dist.Bernoulli(probs[state]))


def meaning(utterance, price):
    return utterance == price


def approx(x, b=None):
    if b is None:
        b = 10
    div = x/b
    rounded = float(int(div)) + 1 if div - float(int(div)) > 0.5 else div
    return b * rounded


qud_fns = {
    "price": lambda state: State(price=state.price, valence="none"),
    "valence": lambda state: State(valence=state.valence, price="none"),
    "priceValence": lambda state: State(price=state.price, valence=state.valence),
    "approxPrice": lambda state: State(price=approx(state.price), valence="none"),
    "approxPriceValence": lambda state: State(price=approx(state.price), valence=state.valence),
}


def qud_prior():
    values = ["price", "valence", "priceValence", "approxPrice", "approxPriceValence"]
    ix = pyro.sample("qud", dist.Categorical(torch.ones(len(values)) / len(values)))
    return values[ix]


def utterance_cost(numberUtt):
    preciseNumberCost = 1
    return approx(numberUtt) if numberUtt == 0 else preciseNumberCost


def utterance_prior():
    utterances = torch.tensor([50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001])
    utteranceProbs = torch.exp(-torch.tensor(utterances))
    ix = pyro.sample("utterance", dist.Categorical(probs=utteranceProbs))
    return utterances[ix]


def literal_listener(utterance, qud):
    state = State(price=price_prior(), valence=valence_prior())
    factor("literal_meaning", 0. if meaning(utterance, state.price) else -10000.)
    return qud_fns[qud](state)


def speaker(qudValue, qud):
    alpha = 1
    utterance = utterance_prior()
    with poutine.scale(scale=torch.tensor(alpha)):
        literal_marginal = HashingMarginal(
            Search(literal_listener).run(utterance, qud))
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
    speaker_marginal = HashingMarginal(Search(speaker).run(qudValue, qud))
    pyro.sample("speaker", speaker_marginal, obs=utterance)
    return state


def main():
    listener_posterior = HashingMarginal(Search(pragmatic_listener).run(10000))
    print(listener_posterior)


if __name__ == "__main__":
    main()
