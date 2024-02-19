Example: using vanilla PyTorch to perform optimization in SVI
=============================================================

This script uses argparse arguments to construct PyTorch optimizer and dataloader, for example::

    $ python examples/svi_torch.py --size 10000 --batch-size 100 --num-epochs 100

`View svi_torch.py on github`__

.. _github: https://github.com/pyro-ppl/pyro/blob/dev/examples/svi_torch.py

__ github_

.. literalinclude:: ../../examples/svi_torch.py
    :language: python
