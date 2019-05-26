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

BetaBinomial
------------
.. autoclass:: pyro.distributions.BetaBinomial
    :members:
    :undoc-members:
    :show-inheritance:

Delta
-----
.. autoclass:: pyro.distributions.Delta
    :members:
    :undoc-members:
    :show-inheritance:

DirichletMultinomial
--------------------
.. autoclass:: pyro.distributions.DirichletMultinomial
    :members:
    :undoc-members:
    :show-inheritance:

EmpiricalDistribution
----------------------
.. autoclass:: pyro.distributions.Empirical
    :members:
    :undoc-members:
    :show-inheritance:

GammaPoisson
------------
.. autoclass:: pyro.distributions.GammaPoisson
    :members:
    :undoc-members:
    :show-inheritance:

GaussianScaleMixture
------------------------------------
.. autoclass:: pyro.distributions.GaussianScaleMixture
    :members:
    :undoc-members:
    :show-inheritance:

InverseGamma
------------
.. autoclass:: pyro.distributions.InverseGamma
    :members:
    :undoc-members:
    :show-inheritance:

LKJCorrCholesky
---------------
.. autoclass:: pyro.distributions.LKJCorrCholesky
    :members:
    :undoc-members:
    :show-inheritance:

MaskedMixture
-------------
.. autoclass:: pyro.distributions.MaskedMixture
    :members:
    :undoc-members:
    :show-inheritance:

MixtureOfDiagNormals
------------------------------------
.. autoclass:: pyro.distributions.MixtureOfDiagNormals
    :members:
    :undoc-members:
    :show-inheritance:

MixtureOfDiagNormalsSharedCovariance
------------------------------------
.. autoclass:: pyro.distributions.MixtureOfDiagNormalsSharedCovariance
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

SpanningTree
------------
.. autoclass:: pyro.distributions.SpanningTree
    :members:
    :undoc-members:
    :show-inheritance:

VonMises
--------
.. autoclass:: pyro.distributions.VonMises
    :members:
    :undoc-members:
    :show-inheritance:

VonMises3D
----------
.. autoclass:: pyro.distributions.VonMises3D
    :members:
    :undoc-members:
    :show-inheritance:

Transformed Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

BatchNormTransform
------------------
.. autoclass:: pyro.distributions.BatchNormTransform
    :members:
    :undoc-members:
    :show-inheritance:

DeepELUFlow
-----------
.. autoclass:: pyro.distributions.DeepELUFlow
    :members:
    :undoc-members:
    :show-inheritance:

DeepLeakyReLUFlow
-----------------
.. autoclass:: pyro.distributions.DeepLeakyReLUFlow
    :members:
    :undoc-members:
    :show-inheritance:

DeepSigmoidalFlow
-----------------
.. autoclass:: pyro.distributions.DeepSigmoidalFlow
    :members:
    :undoc-members:
    :show-inheritance:

HouseholderFlow
---------------
.. autoclass:: pyro.distributions.HouseholderFlow
    :members:
    :undoc-members:
    :show-inheritance:

InverseAutoRegressiveFlow
-------------------------
.. autoclass:: pyro.distributions.InverseAutoregressiveFlow
    :members:
    :undoc-members:
    :show-inheritance:

InverseAutoRegressiveFlowStable
-------------------------------
.. autoclass:: pyro.distributions.InverseAutoregressiveFlowStable
    :members:
    :undoc-members:
    :show-inheritance:

PermuteTransform
----------------
.. autoclass:: pyro.distributions.PermuteTransform
    :members:
    :undoc-members:
    :show-inheritance:

PlanarFlow
----------
.. autoclass:: pyro.distributions.PlanarFlow
    :members:
    :undoc-members:
    :show-inheritance:

RadialFlow
----------
.. autoclass:: pyro.distributions.RadialFlow
    :members:
    :undoc-members:
    :show-inheritance:

SylvesterFlow
-------------
.. autoclass:: pyro.distributions.SylvesterFlow
    :members:
    :undoc-members:
    :show-inheritance:

TransformModule
---------------
.. autoclass:: pyro.distributions.TransformModule
    :members:
    :undoc-members:
    :show-inheritance:
