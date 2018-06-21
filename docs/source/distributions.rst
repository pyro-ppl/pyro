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

.. autoclass:: pyro.distributions.Distribution
    :members:
    :undoc-members:
    :special-members: __call__
    :show-inheritance:

TorchDistributionMixin
----------------------

.. autoclass:: pyro.distributions.torch_distribution.TorchDistributionMixin
    :members:
    :undoc-members:
    :special-members: __call__
    :show-inheritance:
    :member-order: bysource

TorchDistribution
-----------------

.. autoclass:: pyro.distributions.TorchDistribution
    :members:
    :undoc-members:
    :special-members: __call__
    :show-inheritance:
    :member-order: bysource

AVFMultivariateNormal
---------------------
.. autoclass:: pyro.distributions.AVFMultivariateNormal
    :members:
    :undoc-members:
    :show-inheritance:

Binomial
--------

.. autoclass:: pyro.distributions.Binomial
    :members:
    :undoc-members:
    :show-inheritance:

Delta
-----
.. autoclass:: pyro.distributions.Delta
    :members:
    :undoc-members:
    :show-inheritance:

EmpiricalDistribution
----------------------
.. autoclass:: pyro.distributions.Empirical
    :members:
    :undoc-members:
    :show-inheritance:

HalfCauchy
----------
.. autoclass:: pyro.distributions.HalfCauchy
    :members:
    :undoc-members:
    :show-inheritance:

LowRankMultivariateNormal
-------------------------
.. autoclass:: pyro.distributions.LowRankMultivariateNormal
    :members:
    :undoc-members:
    :show-inheritance:

OMTMultivariateNormal
---------------------
.. autoclass:: pyro.distributions.OMTMultivariateNormal
    :members:
    :undoc-members:
    :show-inheritance:

RelaxedBernoulliStraightThrough
-------------------------------
.. autoclass:: pyro.distributions.RelaxedBernoulliStraightThrough
    :members:
    :undoc-members:
    :show-inheritance:

RelaxedOneHotCategoricalStraightThrough
---------------------------------------
.. autoclass:: pyro.distributions.RelaxedOneHotCategoricalStraightThrough
    :members:
    :undoc-members:
    :show-inheritance:

Rejector
--------
.. autoclass:: pyro.distributions.Rejector
    :members:
    :undoc-members:
    :show-inheritance:

VonMises
--------
.. autoclass:: pyro.distributions.VonMises
    :members:
    :undoc-members:
    :show-inheritance:

Transformed Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

InverseAutoRegressiveFlow
-------------------------
.. autoclass:: pyro.distributions.InverseAutoregressiveFlow
    :members:
    :undoc-members:
    :show-inheritance:
