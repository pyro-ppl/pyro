import pytest
import torch

from pyro.infer.mcmc.adaptation import WarmupAdapter, adapt_window
from tests.common import assert_equal


@pytest.mark.parametrize("adapt_step_size, adapt_mass, warmup_steps, expected", [
    (False, False, 100, []),
    (False, True, 50, [(0, 6), (7, 44), (45, 49)]),
    (True, False, 150, [(0, 74), (75, 99), (100, 149)]),
    (True, True, 200, [(0, 74), (75, 99), (100, 149), (150, 199)]),
    (True, True, 280, [(0, 74), (75, 99), (100, 229), (230, 279)]),
    (True, True, 18, [(0, 17)]),
])
def test_adaptation_schedule(adapt_step_size, adapt_mass, warmup_steps, expected):
    adapter = WarmupAdapter(0.1,
                            adapt_step_size=adapt_step_size,
                            adapt_mass_matrix=adapt_mass)
    adapter.configure(warmup_steps, inv_mass_matrix=torch.eye(5, 5))
    expected_schedule = [adapt_window(i, j) for i, j in expected]
    assert_equal(adapter.adaptation_schedule, expected_schedule, prec=0)
