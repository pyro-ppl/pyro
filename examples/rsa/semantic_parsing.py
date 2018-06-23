"""
Combining models of RSA pragmatics and CCG-based compositional semantics.

Taken from: http://dippl.org/examples/zSemanticPragmaticMashup.html
"""

import torch

import argparse
import collections

import pyro
import pyro.distributions as dist

from search_inference import HashingMarginal, BestFirstSearch, factor, memoize

torch.set_default_dtype(torch.float64)


def Marginal(fn=None, **kwargs):
    if fn is None:
        return lambda _fn: Marginal(_fn, **kwargs)
    return memoize(lambda *args: HashingMarginal(BestFirstSearch(fn, **kwargs).run(*args)))


###################################################################
# Lexical semantics
###################################################################

def flip(name, p):
    return pyro.sample(name, dist.Bernoulli(p)).item() == 1


# hashable state
obj = collections.namedtuple("Obj", ["name", "blond", "nice", "tall"])


def Obj(name):
    return obj(name=name,
               blond=flip(name + "_blond", 0.5),
               nice=flip(name + "_nice", 0.5),
               tall=flip(name + "_tall", 0.5))


class Meaning(object):
    def sem(self, world):
        raise NotImplementedError

    __call__ = sem

    def syn(self):
        raise NotImplementedError


class UndefinedMeaning(Meaning):
    def sem(self, world):
        return None

    def syn(self):
        return ""


class BlondMeaning(Meaning):
    def sem(self, world):
        return lambda obj: obj.blond

    def syn(self):
        return {"dir": "L", "int": "NP", "out": "S"}


class NiceMeaning(Meaning):
    def sem(self, world):
        return lambda obj: obj.nice

    def syn(self):
        return {"dir": "L", "int": "NP", "out": "S"}


class TallMeaning(Meaning):
    def sem(self, world):
        return lambda obj: obj.tall

    def syn(self):
        return {"dir": "L", "int": "NP", "out": "S"}


class BobMeaning(Meaning):
    def sem(self, world):
        return list(filter(lambda obj: obj.name == "Bob", world))[0]

    def syn(self):
        return "NP"


class SomeMeaning(Meaning):
    def sem(self, world):
        def f1(P):
            def f2(Q):
                return len(list(filter(Q, filter(P, world)))) > 0
            return f2

        return f1

    def syn(self):
        return {
            "dir": "R",
            "int": {"dir": "L", "int": "NP", "out": "S"},
            "out": {
                "dir": "R",
                "int": {"dir": "L", "int": "NP", "out": "S"},
                "out": "S"
            }
        }


class AllMeaning(Meaning):
    def sem(self, world):
        def f1(P):
            def f2(Q):
                return len(list(filter(lambda *args: not Q(*args),
                                       filter(P, world)))) == 0
            return f2

        return f1

    def syn(self):
        return {
            "dir": "R",
            "int": {"dir": "L", "int": "NP", "out": "S"},
            "out": {
                "dir": "R",
                "int": {"dir": "L", "int": "NP", "out": "S"},
                "out": "S"
            }
        }


class NoneMeaning(Meaning):
    def sem(self, world):
        def f1(P):
            def f2(Q):
                return len(list(filter(Q, filter(P, world)))) == 0
            return f2

        return f1

    def syn(self):
        return {
            "dir": "R",
            "int": {"dir": "L", "int": "NP", "out": "S"},
            "out": {
                "dir": "R",
                "int": {"dir": "L", "int": "NP", "out": "S"},
                "out": "S"
            }
        }


class CompoundMeaning(Meaning):
    def __init__(self, sem, syn):
        self._sem = sem
        self._syn = syn

    def sem(self, world):
        return self._sem(world)

    def syn(self):
        return self._syn


###################################################################
# Compositional semantics
###################################################################

def heuristic(is_good):
    if is_good:
        return torch.tensor(0.)
    return torch.tensor(-100.0)


def world_prior(num_objs, meaning_fn):
    prev_factor = torch.tensor(0.)
    world = []
    for i in range(num_objs):
        world.append(Obj("obj_{}".format(i)))
        new_factor = heuristic(meaning_fn(world))
        factor("factor_{}".format(i), new_factor - prev_factor)
        prev_factor = new_factor

    factor("factor_{}".format(num_objs), prev_factor * -1)
    return tuple(world)


def lexical_meaning(word):
    meanings = {
        "blond": BlondMeaning,
        "nice": NiceMeaning,
        "Bob": BobMeaning,
        "some": SomeMeaning,
        "none": NoneMeaning,
        "all": AllMeaning
    }
    if word in meanings:
        return meanings[word]()
    else:
        return UndefinedMeaning()


def apply_world_passing(f, a):
    return lambda w: f(w)(a(w))


def syntax_match(s, t):
    if "dir" in s and "dir" in t:
        return (s["dir"] and t["dir"]) and \
            syntax_match(s["int"], t["int"]) and \
            syntax_match(s["out"], t["out"])
    else:
        return s == t


def can_apply(meanings):
    inds = []
    for i, meaning in enumerate(meanings):
        applies = False
        s = meaning.syn()
        if "dir" in s:
            if s["dir"] == "L":
                applies = syntax_match(s["int"], meanings[i-1].syn())
            elif s["dir"] == "R":
                applies = syntax_match(s["int"], meanings[i+1].syn())
            else:
                applies = False

        if applies:
            inds.append(i)

    return inds


def combine_meaning(meanings, c):
    possible_combos = can_apply(meanings)
    N = len(possible_combos)
    ix = pyro.sample("ix_{}".format(c),
                     dist.Categorical(torch.ones(N) / N))
    i = possible_combos[ix]
    s = meanings[i].syn()
    if s["dir"] == "L":
        f = meanings[i].sem
        a = meanings[i-1].sem
        new_meaning = CompoundMeaning(sem=apply_world_passing(f, a),
                                      syn=s["out"])
        return meanings[0:i-1] + [new_meaning] + meanings[i+1:]
    if s["dir"] == "R":
        f = meanings[i].sem
        a = meanings[i+1].sem
        new_meaning = CompoundMeaning(sem=apply_world_passing(f, a),
                                      syn=s["out"])
        return meanings[0:i] + [new_meaning] + meanings[i+2:]


def combine_meanings(meanings, c=0):
    if len(meanings) == 1:
        return meanings[0].sem
    else:
        return combine_meanings(combine_meaning(meanings, c), c=c+1)


def meaning(utterance):
    defined = filter(lambda w: "" != w.syn(),
                     list(map(lexical_meaning, utterance.split(" "))))
    return combine_meanings(list(defined))


@Marginal(num_samples=100)
def literal_listener(utterance):
    m = meaning(utterance)
    world = world_prior(2, m)
    factor("world_constraint", heuristic(m(world)) * 1000)
    return world


def utterance_prior():
    utterances = ["some of the blond people are nice",
                  "all of the blond people are nice",
                  "none of the blond people are nice"]
    ix = pyro.sample("utterance", dist.Categorical(torch.ones(3) / 3.0))
    return utterances[ix]


@Marginal(num_samples=100)
def speaker(world):
    utterance = utterance_prior()
    L = literal_listener(utterance)
    pyro.sample("speaker_constraint", L, obs=world)
    return utterance


def rsa_listener(utterance, qud):
    world = world_prior(2, meaning(utterance))
    S = speaker(world)
    pyro.sample("listener_constraint", S, obs=utterance)
    return qud(world)


def literal_listener_raw(utterance, qud):
    m = meaning(utterance)
    world = world_prior(3, m)
    factor("world_constraint", heuristic(m(world)) * 1000)
    return qud(world)


def main(args):

    mll = Marginal(literal_listener_raw, num_samples=args.num_samples)

    def is_any_qud(world):
        return any(map(lambda obj: obj.nice, world))

    print(mll("all blond people are nice", is_any_qud)())

    def is_all_qud(world):
        m = True
        for obj in world:
            if obj.blond:
                if obj.nice:
                    m = m and True
                else:
                    m = m and False
            else:
                m = m and True
        return m

    rsa = Marginal(rsa_listener, num_samples=args.num_samples)

    print(rsa("some of the blond people are nice", is_all_qud)())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=10, type=int)
    args = parser.parse_args()
    main(args)
