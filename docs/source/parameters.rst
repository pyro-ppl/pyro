Parameters
==========

Parameters are variables that are tagged as parameters and stored in the param store in memory.
Parameters can also be saved and loaded from disk.

.. function:: get_param(self, name, init_tensor=None)

  * name: parameter name

  Get parameter from its name. If it does not yet exist in the param store, it will be created and stored.

.. function:: param_name(self, p)

  * p: parameter

  Get parameter name of paramter `p`

.. function:: save(self, filename)

  * filename: name of file on disk to save to. Will overwrite existing file.

  Saves parameters to file.

.. function:: load(self, filename)

  * filename: name of file on disk to load from

  Load parameters from file.
