import torch

import numbers
import collections

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, memoize, Search

torch.set_default_dtype(torch.float64)

# hashable params
Params = collections.namedtuple("Params", ["theta", "gamma", "delta"])


Marginal = lambda fn: memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))
bins = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]


def discretize_beta_pdf(gamma, delta):
    shape_alpha = gamma * delta
    shape_beta = (1.-gamma) * delta
    beta_pdf = lambda x: (x ** (shape_alpha-1)) * ((1.-x)**(shape_beta-1))
    return torch.tensor([beta_pdf(b) for b in bins])


@Marginal
def structured_prior_model(params):
    propertyIsPresent = pyro.sample("propertyIsPresent",
                                    dist.Bernoulli(params.theta)).item() == 1
    if propertyIsPresent:
        ix = pyro.sample("bin", dist.Categorical(probs=discretize_beta_pdf(params.gamma, params.delta)))
        return bins[ix]
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


def main():
    hasWingsERP = structured_prior_model(Params(theta=0.5, gamma=0.99, delta=10.))
    laysEggsERP = structured_prior_model(Params(theta=0.5, gamma=0.5, delta=10.))
    carriesMalariaERP = structured_prior_model(
        Params(theta=0.1, gamma=0.01, delta=2.))
    areFemaleERP = structured_prior_model(Params(theta=0.99, gamma=0.5, delta=50.))


    # listener interpretation of generics
    malariaPosterior = listener1("generic is true", carriesMalariaERP)
    eggsPosterior = listener1("generic is true", laysEggsERP)
    femalePosterior = listener1("generic is true", areFemaleERP)

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
    main()
