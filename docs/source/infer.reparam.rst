Reparameterizers
================
.. automodule:: pyro.infer.reparam

The :mod:`pyro.infer.reparam` module contains reparameterization strategies for
the :func:`pyro.poutine.handlers.reparam` effect. These are useful for altering
geometry of a poorly-conditioned parameter space to make the posterior better
shaped. These can be used with a variety of inference algorithms, e.g.
``Auto*Normal`` guides and MCMC.

.. automodule:: pyro.infer.reparam.reparam
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__

Conjugate Updating
------------------
.. automodule:: pyro.infer.reparam.conjugate
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Loc-Scale Decentering
---------------------
.. automodule:: pyro.infer.reparam.loc_scale
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Transformed Distributions
-------------------------
.. automodule:: pyro.infer.reparam.transform
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Discrete Cosine Transform
-------------------------
.. automodule:: pyro.infer.reparam.discrete_cosine
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Haar Transform
--------------
.. automodule:: pyro.infer.reparam.haar
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Unit Jacobian Transforms
------------------------
.. automodule:: pyro.infer.reparam.unit_jacobian
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

StudentT Distributions
----------------------
.. automodule:: pyro.infer.reparam.studentt
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Stable Distributions
--------------------
.. automodule:: pyro.infer.reparam.stable
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Hidden Markov Models
--------------------
.. automodule:: pyro.infer.reparam.hmm
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Site Splitting
--------------
.. automodule:: pyro.infer.reparam.split
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:

Neural Transport
----------------
.. automodule:: pyro.infer.reparam.neutra
    :members:
    :undoc-members:
    :member-order: bysource
    :special-members: __call__
    :show-inheritance:
