.. Pyro Tutorials documentation master file, created by
   sphinx-quickstart on Tue Oct 31 11:33:17 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pyro Examples and Tutorials!
==========================================

This page collects how-to guides and examples written by the Pyro community.
If you're new to probabilistic programming or variational inference,
you might want to start by reading our :ref:`introductory-tutorials`.

After that, you're ready to get your hands dirty!

A basic familiarity with this introductory material is all you will need
to dive right into using Pyro's two superpowers:
:ref:`deep-generative-models` and :ref:`inference-with-discrete-latent-variables`.

Another feature of Pyro is its programmability, the subject of a series of tutorials in :ref:`customizing-inference`.
Particularly advanced or enthusiastic users may even be interested in :ref:`understanding-pyros-internals`.

.. toctree::
   :maxdepth: 1
   :caption: Introductory Tutorials
   :name: introductory-tutorials

   intro_part_i
   intro_part_ii
   svi_part_i
   svi_part_ii
   svi_part_iii

.. toctree::
   :maxdepth: 1
   :caption: Hands-On Bayesian Modelling
   :name: hands-on-bayesian-modelling

   bayesian_regression
   bayesian_regression_ii
   mle_map

.. toctree::
   :maxdepth: 1
   :caption: Understanding Pyro and PyTorch
   :name: understanding-pyro-and-pytorch

   tensor_shapes
   modules
   jit

.. toctree::
   :maxdepth: 1
   :caption: Deep Generative Models
   :name: deep-generative-models

   vae
   ss-vae
   cvae
   normalizing_flows_i
   dmm
   air
   cevae
   sparse_gamma
   svi_horovod

.. toctree::
   :maxdepth: 1
   :caption: Inference With Discrete Variables
   :name: inference-with-discrete-variables

   enumeration
   gmm
   dirichlet_process_mixture
   toy_mixture_model_discrete_enumeration
   hmm
   capture_recapture
   einsum
   lda

.. toctree::
   :maxdepth: 1
   :caption: Customizing Inference
   :name: customizing-inference

   easyguide
   custom_objectives
   boosting_bbvi
   neutra

.. toctree::
   :maxdepth: 1
   :caption: Application: Time Series
   :name: time-series

   forecasting_i
   forecasting_ii
   forecasting_iii
   forecasting_dlm
   forecast_simple
   stable
   timeseries

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

.. toctree::
   :maxdepth: 1
   :caption: Application: Optimal Experiment Design
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
   :caption: Other Examples:

   RSA-implicature
   RSA-hyperbole
   mcmc
   csis
   smcfilter

.. toctree::
   :maxdepth: 1
   :caption: Developers: Understanding Pyro's Internals
   :name: understanding-pyros-internals

   minipyro
   effect_handlers
   contrib_funsor_intro_i
   contrib_funsor_intro_ii

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
