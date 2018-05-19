Parameters
==========

Parameters in Pyro are basically thin wrappers around PyTorch Tensors that carry unique names. 
As such Parameters are the primary stateful objects in Pyro. Users typically interact with parameters
via the Pyro primitive `pyro.param`. Parameters play a central role in stochastic variational inference,
where they are used to represent point estimates for the parameters in parameterized families of 
models and guides.

.. include:: pyro.params.txt
