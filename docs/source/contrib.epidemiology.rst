Epidemiological Models
======================
.. automodule:: pyro.contrib.epidemiology

.. warning:: Code in ``pyro.contrib.forecast`` is under development.
    This code makes no guarantee about maintaining backwards compatibility.

``pyro.contrib.epidemiology`` is a framework for experimenting with a restricted
class of stochastic discrete-time discrete-count compartmental models. This
framework implements **inference** via Hamiltonian Monte Carlo, **prediction**
of latent variables, and **forecasting** future trajectories.

For explanation of the underlying machinery in this framework, see the
`SIR with HMC tutorial <http://pyro.ai/examples/sir_hcm.html>`_

Base Compartmental Model
------------------------
.. automodule:: pyro.contrib.epidemiology.compartmental
    :members:
    :member-order: bysource

SIR Models
----------
.. automodule:: pyro.contrib.epidemiology.sir
    :members:
    :member-order: bysource
