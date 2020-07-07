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
    :member-order: bysource

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

CoalescentTimes
---------------
.. autoclass:: pyro.distributions.CoalescentTimes
    :members:
    :undoc-members:
    :show-inheritance:

CoalescentTimesWithRate
-----------------------
.. autoclass:: pyro.distributions.CoalescentTimesWithRate
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalDistribution
-----------------------
.. autoclass:: pyro.distributions.ConditionalDistribution
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalTransformedDistribution
----------------------------------
.. autoclass:: pyro.distributions.ConditionalTransformedDistribution
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

DiscreteHMM
-----------
.. autoclass:: pyro.distributions.DiscreteHMM
    :members:
    :undoc-members:
    :show-inheritance:

EmpiricalDistribution
---------------------
.. autoclass:: pyro.distributions.Empirical
    :members:
    :undoc-members:
    :show-inheritance:

ExtendedBetaBinomial
--------------------
.. autoclass:: pyro.distributions.ExtendedBetaBinomial
    :members:
    :undoc-members:
    :show-inheritance:

ExtendedBinomial
----------------
.. autoclass:: pyro.distributions.ExtendedBinomial
    :members:
    :undoc-members:
    :show-inheritance:

FoldedDistribution
---------------------
.. autoclass:: pyro.distributions.FoldedDistribution
    :members:
    :undoc-members:
    :show-inheritance:

GammaGaussianHMM
----------------
.. autoclass:: pyro.distributions.GammaGaussianHMM
    :members:
    :undoc-members:
    :show-inheritance:

GammaPoisson
------------
.. autoclass:: pyro.distributions.GammaPoisson
    :members:
    :undoc-members:
    :show-inheritance:

GaussianHMM
-----------
.. autoclass:: pyro.distributions.GaussianHMM
    :members:
    :undoc-members:
    :show-inheritance:

GaussianMRF
-----------
.. autoclass:: pyro.distributions.GaussianMRF
    :members:
    :undoc-members:
    :show-inheritance:

GaussianScaleMixture
--------------------
.. autoclass:: pyro.distributions.GaussianScaleMixture
    :members:
    :undoc-members:
    :show-inheritance:

ImproperUniform
---------------
.. autoclass:: pyro.distributions.improper_uniform.ImproperUniform
    :members:
    :undoc-members:
    :show-inheritance:

IndependentHMM
--------------
.. autoclass:: pyro.distributions.IndependentHMM
    :members:
    :undoc-members:
    :show-inheritance:

InverseGamma
------------
.. autoclass:: pyro.distributions.InverseGamma
    :members:
    :undoc-members:
    :show-inheritance:

LinearHMM
---------
.. autoclass:: pyro.distributions.LinearHMM
    :members:
    :undoc-members:
    :show-inheritance:

LKJCorrCholesky
---------------
.. autoclass:: pyro.distributions.LKJCorrCholesky
    :members:
    :undoc-members:
    :show-inheritance:

MaskedDistribution
------------------
.. autoclass:: pyro.distributions.MaskedDistribution
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
--------------------
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

MultivariateStudentT
--------------------
.. autoclass:: pyro.distributions.MultivariateStudentT
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

Stable
------
.. autoclass:: pyro.distributions.Stable
    :members:
    :undoc-members:
    :show-inheritance:

TruncatedPolyaGamma
-------------------
.. autoclass:: pyro.distributions.TruncatedPolyaGamma
    :members:
    :undoc-members:
    :show-inheritance:

Unit
----
.. autoclass:: pyro.distributions.Unit
    :members:
    :undoc-members:
    :show-inheritance:

VonMises3D
----------
.. autoclass:: pyro.distributions.VonMises3D
    :members:
    :undoc-members:
    :show-inheritance:

ZeroInflatedPoisson
-------------------
.. autoclass:: pyro.distributions.ZeroInflatedPoisson
    :members:
    :undoc-members:
    :show-inheritance:

ZeroInflatedNegativeBinomial
----------------------------
.. autoclass:: pyro.distributions.ZeroInflatedNegativeBinomial
    :members:
    :undoc-members:
    :show-inheritance:

ZeroInflatedDistribution
------------------------
.. autoclass:: pyro.distributions.ZeroInflatedDistribution
    :members:
    :undoc-members:
    :show-inheritance:

Transforms
~~~~~~~~~~

ConditionalTransform
--------------------
.. autoclass:: pyro.distributions.ConditionalTransform
    :members:
    :undoc-members:
    :show-inheritance:

CorrLCholeskyTransform
----------------------
.. autoclass:: pyro.distributions.transforms.CorrLCholeskyTransform
    :members:
    :undoc-members:
    :show-inheritance:

ELUTransform
------------
.. autoclass:: pyro.distributions.transforms.ELUTransform
    :members:
    :undoc-members:
    :show-inheritance:

HaarTransform
-------------
.. autoclass:: pyro.distributions.transforms.HaarTransform
    :members:
    :undoc-members:
    :show-inheritance:

LeakyReLUTransform
------------------
.. autoclass:: pyro.distributions.transforms.LeakyReLUTransform
    :members:
    :undoc-members:
    :show-inheritance:

LowerCholeskyAffine
-------------------
.. autoclass:: pyro.distributions.transforms.LowerCholeskyAffine
    :members:
    :undoc-members:
    :show-inheritance:

Permute
-------
.. autoclass:: pyro.distributions.transforms.Permute
    :members:
    :undoc-members:
    :show-inheritance:

DiscreteCosineTransform
-----------------------
.. autoclass:: pyro.distributions.transforms.DiscreteCosineTransform
    :members:
    :undoc-members:
    :show-inheritance:

TransformModules
~~~~~~~~~~~~~~~~

AffineAutoregressive
--------------------
.. autoclass:: pyro.distributions.transforms.AffineAutoregressive
    :members:
    :undoc-members:
    :show-inheritance:

AffineCoupling
--------------
.. autoclass:: pyro.distributions.transforms.AffineCoupling
    :members:
    :undoc-members:
    :show-inheritance:

BatchNorm
---------
.. autoclass:: pyro.distributions.transforms.BatchNorm
    :members:
    :undoc-members:
    :show-inheritance:

BlockAutoregressive
-------------------
.. autoclass:: pyro.distributions.transforms.BlockAutoregressive
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalAffineAutoregressive
-------------------------------
.. autoclass:: pyro.distributions.transforms.ConditionalAffineAutoregressive
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalAffineCoupling
-------------------------
.. autoclass:: pyro.distributions.transforms.ConditionalAffineCoupling
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalGeneralizedChannelPermute
------------------------------------
.. autoclass:: pyro.distributions.transforms.ConditionalGeneralizedChannelPermute
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalHouseholder
----------------------
.. autoclass:: pyro.distributions.transforms.ConditionalHouseholder
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalNeuralAutoregressive
-------------------------------
.. autoclass:: pyro.distributions.transforms.ConditionalNeuralAutoregressive
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalPlanar
-----------------
.. autoclass:: pyro.distributions.transforms.ConditionalPlanar
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalRadial
-----------------
.. autoclass:: pyro.distributions.transforms.ConditionalRadial
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalSpline
-----------------
.. autoclass:: pyro.distributions.transforms.ConditionalSpline
    :members:
    :undoc-members:
    :show-inheritance:

ConditionalTransformModule
--------------------------
.. autoclass:: pyro.distributions.ConditionalTransformModule
    :members:
    :undoc-members:
    :show-inheritance:

GeneralizedChannelPermute
-------------------------
.. autoclass:: pyro.distributions.transforms.GeneralizedChannelPermute
    :members:
    :undoc-members:
    :show-inheritance:

Householder
-----------
.. autoclass:: pyro.distributions.transforms.Householder
    :members:
    :undoc-members:
    :show-inheritance:

MatrixExponential
-----------------
.. autoclass:: pyro.distributions.transforms.MatrixExponential
    :members:
    :undoc-members:
    :show-inheritance:

NeuralAutoregressive
--------------------
.. autoclass:: pyro.distributions.transforms.NeuralAutoregressive
    :members:
    :undoc-members:
    :show-inheritance:

Planar
------
.. autoclass:: pyro.distributions.transforms.Planar
    :members:
    :undoc-members:
    :show-inheritance:

Polynomial
----------
.. autoclass:: pyro.distributions.transforms.Polynomial
    :members:
    :undoc-members:
    :show-inheritance:

Radial
------
.. autoclass:: pyro.distributions.transforms.Radial
    :members:
    :undoc-members:
    :show-inheritance:

Spline
------
.. autoclass:: pyro.distributions.transforms.Spline
    :members:
    :undoc-members:
    :show-inheritance:

SplineAutoregressive
--------------------
.. autoclass:: pyro.distributions.transforms.SplineAutoregressive
    :members:
    :undoc-members:
    :show-inheritance:

SplineCoupling
--------------
.. autoclass:: pyro.distributions.transforms.SplineCoupling
    :members:
    :undoc-members:
    :show-inheritance:

Sylvester
---------
.. autoclass:: pyro.distributions.transforms.Sylvester
    :members:
    :undoc-members:
    :show-inheritance:

TransformModule
---------------
.. autoclass:: pyro.distributions.TransformModule
    :members:
    :undoc-members:
    :show-inheritance:

ComposeTransformModule
----------------------
.. autoclass:: pyro.distributions.ComposeTransformModule
    :members:
    :undoc-members:
    :show-inheritance:

Transform Factories
~~~~~~~~~~~~~~~~~~~

Each :class:`~torch.distributions.transforms.Transform` and :class:`~pyro.distributions.TransformModule` includes a corresponding helper function in lower case that inputs, at minimum, the input dimensions of the transform, and possibly additional arguments to customize the transform in an intuitive way. The purpose of these helper functions is to hide from the user whether or not the transform requires the construction of a hypernet, and if so, the input and output dimensions of that hypernet.


iterated
--------
.. autofunction:: pyro.distributions.transforms.iterated

affine_autoregressive
---------------------
.. autofunction:: pyro.distributions.transforms.affine_autoregressive

affine_coupling
---------------
.. autofunction:: pyro.distributions.transforms.affine_coupling

batchnorm
---------
.. autofunction:: pyro.distributions.transforms.batchnorm

block_autoregressive
--------------------
.. autofunction:: pyro.distributions.transforms.block_autoregressive

conditional_affine_autoregressive
---------------------------------
.. autofunction:: pyro.distributions.transforms.conditional_affine_autoregressive

conditional_affine_coupling
---------------------------
.. autofunction:: pyro.distributions.transforms.conditional_affine_coupling

conditional_generalized_channel_permute
---------------------------------------
.. autofunction:: pyro.distributions.transforms.conditional_generalized_channel_permute

conditional_householder
-----------------------
.. autofunction:: pyro.distributions.transforms.conditional_householder

conditional_neural_autoregressive
---------------------------------
.. autofunction:: pyro.distributions.transforms.conditional_neural_autoregressive

conditional_planar
------------------
.. autofunction:: pyro.distributions.transforms.conditional_planar

conditional_radial
------------------
.. autofunction:: pyro.distributions.transforms.conditional_radial

conditional_spline
------------------
.. autofunction:: pyro.distributions.transforms.conditional_spline

elu
---
.. autofunction:: pyro.distributions.transforms.elu

generalized_channel_permute
---------------------------
.. autofunction:: pyro.distributions.transforms.generalized_channel_permute

householder
-----------
.. autofunction:: pyro.distributions.transforms.householder

leaky_relu
----------
.. autofunction:: pyro.distributions.transforms.leaky_relu

matrix_exponential
------------------
.. autofunction:: pyro.distributions.transforms.matrix_exponential

neural_autoregressive
---------------------
.. autofunction:: pyro.distributions.transforms.neural_autoregressive

permute
-------
.. autofunction:: pyro.distributions.transforms.permute

planar
------
.. autofunction:: pyro.distributions.transforms.planar

polynomial
----------
.. autofunction:: pyro.distributions.transforms.polynomial

radial
------
.. autofunction:: pyro.distributions.transforms.radial

spline
------
.. autofunction:: pyro.distributions.transforms.spline

spline_autoregressive
---------------------
.. autofunction:: pyro.distributions.transforms.spline_autoregressive

spline_coupling
---------------
.. autofunction:: pyro.distributions.transforms.spline_coupling

sylvester
---------
.. autofunction:: pyro.distributions.transforms.sylvester
