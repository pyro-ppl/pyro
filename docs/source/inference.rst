Inference
=========

In the context of probabilistic modeling, learning is usually called inference.
In the case where the model has latent random variables, this involves computing
(approximate) posterior distributions. In the case of parameterized models, this
usually involves some sort of optimization. Pyro supports multiple inference algorithms,
with support for stochastic variational inference being the most extensive. 
Look here for more inference algorithms in future versions of Pyro.

.. include:: pyro.infer.txt
