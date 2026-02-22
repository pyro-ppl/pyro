.. Pyro Tutorials documentation master file, created by
   sphinx-quickstart on Tue Oct 31 11:33:17 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting Started With Pyro: Tutorials, How-to Guides and Examples
================================================================

Welcome! This page collects tutorials written by the Pyro community.
If you're having trouble finding or understanding anything here,
please don't hesitate to ask a question on our `forum <https://forum.pyro.ai/>`_!

New users: getting from zero to one
------------------------------------
If you're new to probabilistic programming or variational inference,
you might want to start by reading the series :ref:`introductory-tutorials`, especially the :doc:`Introduction to Pyro <intro_long>`.
If you're new to PyTorch, you may also benefit from reading the official introduction `"Deep Learning with PyTorch." <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>`_

After that, you're ready to get started using Pyro! (Yes, really!)
Follow `the instructions on the front page to install Pyro <http://pyro.ai/#install>`_
and look carefully through the series :ref:`practical-pyro-and-pytorch`,
especially the :doc:`first Bayesian regression tutorial <bayesian_regression>`.
This tutorial goes step-by-step through solving a simple Bayesian machine learning problem with Pyro,
grounding the concepts from the introductory tutorials in runnable code.
Users interested in integrating with existing PyTorch training and serving infrastructure should also read :doc:`the PyroModule tutorial <modules>`
and look at the :doc:`SVI with PyTorch <svi_torch>` and :doc:`SVI with Lightning <svi_lightning>` examples.

Most users who reach this point will also find our :doc:`guide to tensor shapes in Pyro <tensor_shapes>` essential reading.
Pyro makes extensive use of the behavior of `"array broadcasting" <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
baked into PyTorch and other array libraries to parallelize models and inference algorithms,
and while it can be difficult to understand this behavior initially, applying the intuition and rules of thumb there
will go a long way toward making your experience smooth and avoiding nasty shape errors.

Core functionality: Deep learning, discrete variables and customizable inference
---------------------------------------------------------------------------------
A basic familiarity with this introductory material is all you will need to dive right into exploiting Pyro's two biggest strengths:
integration with deep learning and automated exact inference for discrete latent variables.
The former is described with numerous examples in the series :ref:`deep-generative-models`.
All are elaborations on the basic idea of the variational autoencoder, introduced in great detail in :doc:`the first tutorial of this series <vae>`.

Pyro's facility with discrete latent variable models like the hidden Markov model is surveyed in the series :ref:`discrete-latent-variables`.
Making use of this in your own work will require careful reading of :doc:`our overview and programming guide <enumeration>` that opens this series.

Another feature of Pyro is its programmability, the subject of a series of tutorials in :ref:`customizing-inference`.
Users working with large models where only part of the model needs special attention
may be interested in `pyro.contrib.easyguide <http://docs.pyro.ai/en/dev/contrib.easyguide.html>`_, introduced in :doc:`the first tutorial of the series <easyguide>`.
Meanwhile, machine learning researchers interested in developing variational inference algorithms may wish to peruse
:doc:`the guide to implementing custom variational objectives <custom_objectives>`,
and a companion example that walks through :doc:`implementing "Boosting BBVI" <boosting_bbvi>`.

Particularly enthusiastic users and potential contributors, especially those interested in contributing to Pyro's core components,
may even be interested in how Pyro itself works under the hood, partially described in the series :ref:`understanding-pyros-internals`.
The :doc:`mini-pyro example <minipyro>` contains a complete and heavily commented implementation of a small version of the Pyro language in just a few hundred lines of code,
and should serve as a more digestable introduction to the real thing.

Tools for specific problems
-----------------------------
Pyro is a mature piece of open-source software with "batteries included."
In addition to the core machinery for modelling and inference,
it includes a large toolkit of dedicated domain- or problem-specific modelling functionality.

One particular area of strength is time-series modelling via `pyro.contrib.forecasting <http://docs.pyro.ai/en/dev/contrib.forecast.html>`_,
a library for scaling hierarchical, fully Bayesian models of multivariate time series to thousands or millions of series and datapoints.
This is described in the series :ref:`time-series`.

Another area of strength is probabilistic machine learning with Gaussian processes.
`pyro.contrib.gp <http://docs.pyro.ai/en/dev/contrib.gp.html>`_, described in the series :ref:`gaussian-processes`,
is a library within Pyro implementing a variety of exact or approximate Gaussian process models compatible with Pyro's inference engines.
Pyro is also fully compatible with `GPyTorch <https://gpytorch.ai/>`_, a dedicated library for scalable GPs,
as described in `their Pyro example series <https://github.com/cornellius-gp/gpytorch/tree/master/examples/07_Pyro_Integration>`_.

List of Tutorials
==================

.. toctree::
   :maxdepth: 1
   :caption: Introductory Tutorials
   :name: introductory-tutorials

   intro_long
   model_rendering
   svi_part_i
   svi_part_ii
   svi_part_iii
   svi_part_iv

.. toctree::
   :maxdepth: 1
   :caption: Practical Pyro and PyTorch
   :name: practical-pyro-and-pytorch

   bayesian_regression
   bayesian_regression_ii
   tensor_shapes
   modules
   workflow
   prior_predictive
   jit
   svi_torch
   svi_horovod
   svi_lightning
   svi_flow_guide

.. toctree::
   :maxdepth: 1
   :caption: Deep Generative Models
   :name: deep-generative-models

   vae
   ss-vae
   cvae
   normalizing_flows_intro
   vae_flow_prior
   dmm
   air
   cevae
   sparse_gamma
   prodlda
   scanvi

.. toctree::
   :maxdepth: 1
   :caption: Discrete Latent Variables
   :name: discrete-latent-variables

   enumeration
   gmm
   dirichlet_process_mixture
   toy_mixture_model_discrete_enumeration
   hmm
   capture_recapture
   mixed_hmm
   einsum
   lda

.. toctree::
   :maxdepth: 1
   :caption: Customizing Inference
   :name: customizing-inference

   mle_map
   easyguide
   custom_objectives
   boosting_bbvi
   neutra
   sparse_regression
   autoname_examples

.. toctree::
   :maxdepth: 1
   :caption: Application: Time Series
   :name: time-series

   forecasting_i
   forecasting_ii
   forecasting_iii
   forecasting_dlm
   stable
   forecast_simple
   timeseries
   reconciling_experts

.. toctree::
   :maxdepth: 1
   :caption: Application: Gaussian Processes
   :name: gaussian-processes

   gp
   gplvm
   bo
   dkl

.. toctree::
   :maxdepth: 1
   :caption: Application: Epidemiology
   :name: epidemiology

   epi_intro
   epi_sir
   epi_regional
   sir_hmc
   logistic-growth

.. toctree::
   :maxdepth: 1
   :caption: Application: Biological sequences
   :name: biological-sequences

   mue_profile
   mue_factor

.. toctree::
   :maxdepth: 1
   :caption: Application: Experimental Design
   :name: optimal-experiment-design

   working_memory
   elections

.. toctree::
   :maxdepth: 1
   :caption: Application: Object Tracking
   :name: object-tracking

   tracking_1d
   ekf

.. toctree::
   :maxdepth: 1
   :caption: Other Inference Algorithms

   baseball
   mcmc
   lkj
   csis
   smcfilter
   inclined_plane
   RSA-implicature
   RSA-hyperbole
   predictive_deterministic

.. toctree::
   :maxdepth: 1
   :caption: Understanding Pyro's Internals
   :name: understanding-pyros-internals

   minipyro
   effect_handlers
   contrib_funsor_intro_i
   contrib_funsor_intro_ii
   hmm_funsor

.. toctree::
   :maxdepth: 1
   :caption: Deprecated
   :name: deprecated

   intro_part_i
   intro_part_ii

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
