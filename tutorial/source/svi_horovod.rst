Distributed training via Horovod
================================

Unlike other examples, this example must be run under horovodrun_, for example

.. _horovodrun: https://github.com/horovod/horovod/blob/master/docs/running.rst

.. code-block:: none

    $ horovodrun -np 2 examples/svi_horovod.py

`View svi_horovod.py on github`__

.. _github: https://github.com/pyro-ppl/pyro/blob/dev/examples/svi_horovod.py

__ github_

.. literalinclude:: ../../examples/svi_horovod.py
    :language: python

