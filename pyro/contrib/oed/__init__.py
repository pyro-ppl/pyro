# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tasks such as choosing the next question to ask in a psychology study, designing an election polling
strategy, and deciding which compounds to synthesize and test in biological sciences are all fundamentally asking the
same question: how do we design an experiment to maximize the information gathered?  Pyro is designed to support
automated optimal experiment design: specifying a model and guide is enough to obtain optimal designs for many different
kinds of experiment scenarios. Check out our experimental design tutorials that use Pyro to
`design an adaptive psychology study <https://pyro.ai/examples/working_memory.html>`_ that uses past data to
select the next question, and `design an election polling strategy <https://pyro.ai/examples/elections.html>`_ that
aims to give the strongest prediction about the eventual winner of the election.

Bayesian optimal experimental design (BOED) is a powerful methodology for tackling experimental design problems and
is the framework adopted by Pyro.
In the BOED framework, we begin with a Bayesian model with a likelihood :math:`p(y|\\theta,d)` and a prior
:math:`p(\\theta)` on the target latent variables. In Pyro, any fully Bayesian model can be used in the BOED framework.
The sample sites corresponding to experimental outcomes are the *observation* sites, those corresponding to
latent variables of interest are the *target* sites. The design :math:`d` is the argument to the model, and is not
a random variable.

In the BOED framework, we choose the design that optimizes the expected information gain (EIG) on the targets
:math:`\\theta` from running the experiment

    :math:`\\text{EIG}(d) = \\mathbf{E}_{p(y|d)} [H[p(\\theta)] − H[p(\\theta|y, d)]]` ,

where :math:`H[·]` represents the entropy and :math:`p(\\theta|y, d) \\propto p(\\theta)p(y|\\theta, d)` is the
posterior we get from
running the experiment with design :math:`d` and observing :math:`y`. In other words, the optimal design is the one
that, in expectation over possible future observations, most reduces posterior entropy
over the target latent variables. If the predictive model is correct, this forms a design strategy that is
(one-step) optimal from an information-theoretic viewpoint. For further details, see [1, 2].

The :mod:`pyro.contrib.oed` module provides tools to create optimal experimental
designs for Pyro models. In particular, it provides estimators for the
expected information gain (EIG).

To estimate the EIG for a particular design, we first set up our Pyro model. For example::

    def model(design):

        # This line allows batching of designs, treating all batch dimensions as independent
        with pyro.plate_stack("plate_stack", design.shape):

            # We use a Normal prior for theta
            theta = pyro.sample("theta", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))

            # We use a simple logistic regression model for the likelihood
            logit_p = theta - design
            y = pyro.sample("y", dist.Bernoulli(logits=logit_p))

            return y

We then select an appropriate EIG estimator, such as::

    eig = nmc_eig(model, design, observation_labels=["y"], target_labels=["theta"], N=2500, M=50)

It is possible to estimate the EIG across a grid of designs::

    designs = torch.stack([design1, design2], dim=0)

to find the best design from a number of options.

[1] Chaloner, Kathryn, and Isabella Verdinelli. "Bayesian experimental design: A review."
Statistical Science (1995): 273-304.

[2] Foster, Adam, et al. "Variational Bayesian Optimal Experimental Design." arXiv preprint arXiv:1903.05480 (2019).

"""

from pyro.contrib.oed import search, eig

__all__ = [
    "search",
    "eig"
]
