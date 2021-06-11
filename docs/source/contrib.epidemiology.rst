Epidemiology
============
.. automodule:: pyro.contrib.epidemiology

.. warning:: Code in ``pyro.contrib.epidemiology`` is under development.
    This code makes no guarantee about maintaining backwards compatibility.

``pyro.contrib.epidemiology`` provides a modeling language for a class of
stochastic discrete-time discrete-count compartmental models. This module
implements black-box **inference** (both Stochastic Variational Inference and
Hamiltonian Monte Carlo), **prediction** of latent variables, and
**forecasting** of future trajectories.

For example usage see the following tutorials:

- `Introduction <http://pyro.ai/examples/epi_intro.html>`_
- `Univariate models <http://pyro.ai/examples/epi_sir.html>`_
- `Regional models <http://pyro.ai/examples/epi_regional.html>`_
- `Inference via auxiliary variable HMC <http://pyro.ai/examples/sir_hmc.html>`_

Base Compartmental Model
------------------------
.. automodule:: pyro.contrib.epidemiology.compartmental
    :members:
    :show-inheritance:
    :member-order: bysource

Example Models
--------------
.. automodule:: pyro.contrib.epidemiology.models

Distributions
-------------
.. automodule:: pyro.contrib.epidemiology.distributions
    :members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyro.distributions.CoalescentRateLikelihood
    :members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__

.. autofunction:: pyro.distributions.coalescent.bio_phylo_to_times
