Epidemiology
============
.. automodule:: pyro.contrib.epidemiology

.. warning:: Code in ``pyro.contrib.epidemiology`` is under development.
    This code makes no guarantee about maintaining backwards compatibility.

``pyro.contrib.epidemiology`` is a framework for experimenting with a
restricted class of stochastic discrete-time discrete-count compartmental
models. This framework implements **inference** (both Stochastic Variational
Inference and Hamiltonian Monte Carlo), **prediction** of latent variables, and
**forecasting** of future trajectories.

For example usage see the `SIR tuorial <http://pyro.ai/examples/contrib/epidemiology/sir.html>`_ .

For explanation of the underlying inference machinery in this framework, see the
`low-level SIR with HMC tutorial <http://pyro.ai/examples/sir_hmc.html>`_ .

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
