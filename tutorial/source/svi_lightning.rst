Example: distributed training via PyTorch Lightning
===================================================

This script passes argparse arguments to PyTorch Lightning ``Trainer`` automatically_, for example::

    $ python examples/svi_lightning.py --accelerator gpu --devices 2 --max_epochs 100 --strategy ddp

.. _automatically: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-in-python-scripts

`View svi_lightning.py on github`__

.. _github: https://github.com/pyro-ppl/pyro/blob/dev/examples/svi_lightning.py

__ github_

.. literalinclude:: ../../examples/svi_lightning.py
    :language: python
