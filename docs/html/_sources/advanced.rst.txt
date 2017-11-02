Advanced Features
=================

Beneath the built-in inference algorithms, Pyro has a library of flexible
primitives for creating new inference algorithms and working with probabilistic
programs.  The core abstraction of composable inference in Pyro is the
`poutine` (short for Pyro Coroutine).  Pyro's inference algorithms are all
built by applying `poutine` s to stochastic functions.

.. toctree::
   :maxdepth: 2

   poutine
