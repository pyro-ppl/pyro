Inference
=========

In the context of probabilistic modeling, learning is usually called inference.
In the particular case of Bayesian inference, this often involves computing
(approximate) posterior distributions. In the case of parameterized models, this
usually involves some sort of optimization. Pyro supports multiple inference algorithms,
with support for stochastic variational inference (SVI) being the most extensive. 
Look here for more inference algorithms in future versions of Pyro.

See `Intro II <http://pyro.ai/examples/intro_part_ii.html>`_ for a discussion of inference in Pyro.

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:

   inference_algos
   mcmc