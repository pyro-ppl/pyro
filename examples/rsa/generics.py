"""
Interpreting generic statements with RSA models of pragmatics.

Taken from:
[0] http://forestdb.org/models/generics.html
[1] https://gscontras.github.io/probLang/chapters/07-generics.html
"""

import torch

import argparse
import numbers
import collections

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, memoize, Search

torch.set_default_dtype(torch.float64)  # double precision for numerical stability


def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))


#######################
# models
#######################

# hashable params
Params = collections.namedtuple("Params", ["theta", "gamma", "delta"])


def discretize_beta_pdf(bins, gamma, delta):
    """
    discretized version of the Beta pdf used for approximately integrating via Search
    """
    shape_alpha = gamma * delta
    shape_beta = (1.-gamma) * delta
    return torch.tensor(
        list(map(lambda x: (x ** (shape_alpha-1)) * ((1.-x)**(shape_beta-1)), bins)))


@Marginal
def structured_prior_model(params):
    propertyIsPresent = pyro.sample("propertyIsPresent",
                                    dist.Bernoulli(params.theta)).item() == 1
    if propertyIsPresent:
        # approximately integrate over a beta by enumerating over bins
        beta_bins = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        ix = pyro.sample("bin", dist.Categorical(
            probs=discretize_beta_pdf(beta_bins, params.gamma, params.delta)))
        return beta_bins[ix]
    else:
        return 0


def threshold_prior():
    threshold_bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ix = pyro.sample("threshold", dist.Categorical(logits=torch.zeros(len(threshold_bins))))
    return threshold_bins[ix]


def utterance_prior():
    utterances = ["generic is true", "mu"]
    ix = pyro.sample("utterance", dist.Categorical(logits=torch.zeros(len(utterances))))
    return utterances[ix]


def meaning(utterance, state, threshold):
    if isinstance(utterance, numbers.Number):
        return state == utterance
    if utterance == "generic is true":
        return state > threshold
    if utterance == "generic is false":
        return state <= threshold
    if utterance == "mu":
        return True
    if utterance == "some":
        return state > 0
    if utterance == "most":
        return state >= 0.5
    if utterance == "all":
        return state >= 0.99
    return True


@Marginal
def listener0(utterance, threshold, prior):
    state = pyro.sample("state", prior)
    m = meaning(utterance, state, threshold)
    factor("listener0_true", 0. if m else -99999.)
    return state


@Marginal
def speaker1(state, threshold, prior):
    s1Optimality = 5.
    utterance = utterance_prior()
    L0 = listener0(utterance, threshold, prior)
    with poutine.scale(scale=torch.tensor(s1Optimality)):
        pyro.sample("L0_score", L0, obs=state)
    return utterance


@Marginal
def listener1(utterance, prior):
    state = pyro.sample("state", prior)
    threshold = threshold_prior()
    S1 = speaker1(state, threshold, prior)
    pyro.sample("S1_score", S1, obs=utterance)
    return state


@Marginal
def speaker2(prevalence, prior):
    utterance = utterance_prior()
    wL1 = listener1(utterance, prior)
    pyro.sample("wL1_score", wL1, obs=prevalence)
    return utterance


def main(args):
    hasWingsERP = structured_prior_model(Params(theta=0.5, gamma=0.99, delta=10.))
    laysEggsERP = structured_prior_model(Params(theta=0.5, gamma=0.5, delta=10.))
    carriesMalariaERP = structured_prior_model(Params(theta=0.1, gamma=0.01, delta=2.))
    areFemaleERP = structured_prior_model(Params(theta=0.99, gamma=0.5, delta=50.))

    # listener interpretation of generics
    wingsPosterior = listener1("generic is true", hasWingsERP)
    malariaPosterior = listener1("generic is true", carriesMalariaERP)
    eggsPosterior = listener1("generic is true", laysEggsERP)
    femalePosterior = listener1("generic is true", areFemaleERP)
    listeners = {"wings": wingsPosterior, "malaria": malariaPosterior,
                 "eggs": eggsPosterior, "female": femalePosterior}

    for name, listener in listeners.items():
        for elt in listener.enumerate_support():
            print(name, elt, listener.log_prob(elt).exp().item())

    # truth judgments
    malariaSpeaker = speaker2(0.1, carriesMalariaERP)
    eggSpeaker = speaker2(0.6, laysEggsERP)
    femaleSpeaker = speaker2(0.5, areFemaleERP)
    lionSpeaker = speaker2(0.01, laysEggsERP)
    speakers = {"malaria": malariaSpeaker, "egg": eggSpeaker,
                "female": femaleSpeaker, "lion": lionSpeaker}

    for name, speaker in speakers.items():
        for elt in speaker.enumerate_support():
            print(name, elt, speaker.log_prob(elt).exp().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=10, type=int)
    args = parser.parse_args()
    main(args)
