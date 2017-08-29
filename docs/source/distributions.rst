Distributions
=============

Distributions in Pyro are represented as primitive objects as well as stochastic functions.
They can be instantiated as objects::

  d = dist.Binomial(p[, batch_size=1])

or treated as functions that take parameters at runtime:: 

  d = dist.binomial(p) // returns a sample of size size(p)

Distributions provide the following methods:

.. function::  .sample([params])

  returns a sample from the parameterized distribution with the same dimensions
  as the parameters.

.. function:: .log_pdf(sample, [params])

  returns the log probability of a sample with the same dimensions
  as the parameters.
  
.. function:: .batch_log_pdf(sample, [params, batch_size=1])

  returns a vectorized score with the same dimensions as the parameters.
  If ``batch_size != 1``. returns a score of size ``[batch_size, params.size()]``

.. function:: .support([params])

  implemented for discrete distributions only. returns an iterator
  over the support of the parameterized distribution.

.. note::
  Parameters should be of type torch ``Variable`` and all methods return type ``Variable``
  unless otherwise noted.

Take a look at the examples[link] to see how they interact with inference algorithms.

Primitives
----------
.. include:: pyro.distributions.txt
