from collections import OrderedDict

import torch

from pyro.contrib.funsor import to_funsor, to_data, named


def test_iteration():
    from funsor.domains import bint, reals
    from funsor.tensor import Tensor

    @named
    def testing():
        for i in named(range(5)):
            v1 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{i}", bint(2))]), 'real'))
            v2 = to_data(Tensor(torch.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
            fv1 = to_funsor(v1, reals())
            fv2 = to_funsor(v2, reals())
            print(i, v1.shape)  # shapes should alternate
            if i % 2 == 0:
                assert v1.shape == (2,)
            else:
                assert v1.shape == (2, 1, 1)
            assert v2.shape == (2, 1)
            print(i, fv1.inputs)
            print('a', v2.shape)  # shapes should stay the same
            print('a', fv2.inputs)

    testing()


def test_nesting():
    from funsor.domains import bint
    from funsor.tensor import Tensor

    @named
    def testing():

        with named():
            v1 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{1}", bint(2))]), 'real'))
            print(1, v1.shape)  # shapes should alternate
            assert v1.shape == (2,)

            with named():
                v2 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{2}", bint(2))]), 'real'))
                print(2, v2.shape)  # shapes should alternate
                assert v2.shape == (2, 1)

                with named():
                    v3 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{3}", bint(2))]), 'real'))
                    print(3, v3.shape)  # shapes should alternate
                    assert v3.shape == (2,)

                    with named():
                        v4 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{4}", bint(2))]), 'real'))
                        print(4, v4.shape)  # shapes should alternate

                        assert v4.shape == (2, 1)

    testing()


def test_staggered():
    from funsor.domains import bint, reals
    from funsor.tensor import Tensor

    @named
    def testing():
        for i in named(range(12)):
            if i % 4 == 0:
                v2 = to_data(Tensor(torch.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
                fv2 = to_funsor(v2, reals())
                assert v2.shape == (2,)
                print('a', v2.shape)  # shapes should also alternate?
                print('a', fv2.inputs)

    testing()
