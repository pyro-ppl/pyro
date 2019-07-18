import pytest
import torch

from pyro.ops.gaussian import Gaussian, gaussian_contract


def random_gaussian(batch_shape, dim, rank):
    """
    Generate a random Gaussian for testing.
    """
    log_normalizer = torch.randn(batch_shape)
    mean = torch.randn(batch_shape + (dim,))
    samples = torch.randn(batch_shape + (dim, rank))
    precision = torch.matmul(samples, samples.transpose(-1, -1))
    result = Gaussian(log_normalizer, mean, precision)
    assert result.dim() == dim
    assert result.batch_shape == batch_shape
    return result


@pytest.mark.parametrize("x_batch_shape,y_batch_shape", [
    ((), ()),
    ((3,), ()),
    ((), (3,)),
    ((2, 1), (3,)),
    ((2, 3), (2, 3,)),
], ids=str)
@pytest.mark.parametrize("equation", [
    "a,ab->b",
    "a,abc->bc",
    "ab,abc->c",
    "ac,abcd->bd",
    "ab,bc->ac",
    "abc,bcd->ad",
    "abcde,bdfgh->bd",
])
@pytest.mark.parametrize("x_rank,y_rank", [
    (1, 1), (4, 1), (1, 4), (4, 4)
], ids=str)
def test_gaussian_contract(equation,
                           x_batch_shape, x_rank,
                           y_batch_shape, y_rank):
    inputs, output = equation.split("->")
    x_input, y_input = inputs.split(",")
    x = random_gaussian(x_batch_shape, len(x_input), x_rank)
    y = random_gaussian(y_batch_shape, len(y_input), y_rank)

    gaussian_contract(equation, x, y)
    # TODO(fehiepsi) Design a test for this.
    # One thing we could do is test against funsor in case that is installed.
