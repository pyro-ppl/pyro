Primitives
==========

Pyro offers operations like sampling and observing as primitives. These functions calls require a unique name.

.. function::  param(name, ...)

  * name: name of parameter

  Saves the variable as a parameter in the param store.  To interact with the param store or write to disk, see `Parameters <parameters.html>`_.

.. function:: sample(name, fn, ...)

  * name: name of sample
  * fn: distribution class or function

  Samples from the distribution and registers it in the trace data structure.

.. function:: observe(name, fn, obs, ...)

  * name: name of observation
  * fn: distribution class or function
  * obs: observed datum

  Only should be used in the context of inference. Calculates the score of the sample and registers it in the trace data structure.

.. function:: map_data(name, data, observer, ...)

  * name: name of mapdata 
  * data: data to subsample 
  * observer: observe function 

  Data subsampling with the important property that all the data are conditionally independent.

.. function:: module(pyro_name, nn_obj, ...)

  * name: name of module
  * nn_obj: pytorch nn module

  Registers the parameters of a pytorch nn module with the param store.
