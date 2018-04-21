Distributions
=============

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:

PyTorch Distributions
~~~~~~~~~~~~~~~~~~~~~

Most distributions in Pyro are thin wrappers around PyTorch distributions.
For details on the PyTorch distribution interface, see
:class:`torch.distributions.distribution.Distribution`.
For differences between the Pyro and PyTorch interfaces, see
:class:`~pyro.distributions.torch_distribution.TorchDistributionMixin`.

.. automodule:: pyro.distributions.torch

Pyro Distributions
~~~~~~~~~~~~~~~~~~

Abstract Distribution
---------------------

.. automodule:: pyro.distributions.distribution
    :members:
    :undoc-members:
    :special-members: __call__
    :show-inheritance:

TorchDistribution
-----------------

.. automodule:: pyro.distributions.torch_distribution
    :members:
    :undoc-members:
    :special-members: __call__
    :show-inheritance:
    :member-order: bysource

Binomial
--------

.. automodule:: pyro.distributions.binomial
    :members:
    :undoc-members:
    :show-inheritance:

Delta
-----
.. automodule:: pyro.distributions.delta
    :members:
    :undoc-members:
    :show-inheritance:

EmpiricalDistribution
----------------------
.. automodule:: pyro.distributions.empirical
    :members:
    :undoc-members:
    :show-inheritance:

HalfCauchy
----------
.. automodule:: pyro.distributions.half_cauchy
    :members:
    :undoc-members:
    :show-inheritance:

LowRankMultivariateNormal
-------------------------
.. automodule:: pyro.distributions.lowrank_mvn
    :members:
    :undoc-members:
    :show-inheritance:

OMTMultivariateNormal
---------------------
.. automodule:: pyro.distributions.omt_mvn
    :members:
    :undoc-members:
    :show-inheritance:

VonMises
--------
.. automodule:: pyro.distributions.von_mises
    :members:
    :undoc-members:
    :show-inheritance:

Transformed Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

InverseAutoRegressiveFlow
-------------------------
.. autoclass:: pyro.distributions.iaf.InverseAutoregressiveFlow
    :members:
    :undoc-members:
    :show-inheritance:
