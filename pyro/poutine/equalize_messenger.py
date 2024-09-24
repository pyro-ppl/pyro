# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import List, Optional, Union

from typing_extensions import Self

from pyro.distributions import Delta
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import Message


class EqualizeMessenger(Messenger):
    """
    Given a stochastic function with some primitive statements and a list of names,
    force the primitive statements at those names to have the same value,
    with that value being the result of the first primitive statement matching those names.

    Consider the following Pyro program:

        >>> def per_category_model(category):
        ...     shift = pyro.param(f'{category}_shift', torch.randn(1))
        ...     mean = pyro.sample(f'{category}_mean', pyro.distributions.Normal(0, 1))
        ...     std = pyro.sample(f'{category}_std', pyro.distributions.LogNormal(0, 1))
        ...     return pyro.sample(f'{category}_values', pyro.distributions.Normal(mean + shift, std))

    Running the program for multiple categories can be done by

        >>> def model(categories):
        ...     return {category:per_category_model(category) for category in categories}

    To make the `std` sample sites have the same value, we can write

        >>> equal_std_model = pyro.poutine.equalize(model, '.+_std')

    If on top of the above we would like to make the 'shift' parameters identical, we can write

        >>> equal_std_param_model = pyro.poutine.equalize(equal_std_model, '.+_shift', 'param')

    Alternatively, the ``equalize``  messenger can be used to condition a model on primitive statements
    having the same value by setting `keep_dist` to True. Consider the below model:

        >>> def model():
        ...     x = pyro.sample('x', pyro.distributions.Normal(0, 1))
        ...     y = pyro.sample('y', pyro.distributions.Normal(5, 3))
        ...     return x, y

    The model can be conditioned on 'x' and 'y' having the same value by

        >>> conditioned_model = pyro.poutine.equalize(model, ['x', 'y'], keep_dist=True)

    Note that the conditioned model defined above calculates the correct unnormalized log-probablity
    density, but in order to correctly sample from it one must use SVI or MCMC techniques.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param sites: a string or list of strings to match site names (the strings can be regular expressions)
    :param type: a string specifying the site type (default is 'sample')
    :param bool keep_dist: Whether to keep the distributions of the second and subsequent
        matching primitive statements. If kept this is equivalent to conditioning the model
        on all matching primitive statements having the same value, as opposed to having the
        second and subsequent matching primitive statements replaced by delta sampling functions.
        Defaults to False.
    :returns: stochastic function decorated with a :class:`~pyro.poutine.equalize_messenger.EqualizeMessenger`
    """

    def __init__(
        self,
        sites: Union[str, List[str]],
        type: Optional[str] = "sample",
        keep_dist: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.sites = [sites] if isinstance(sites, str) else sites
        self.type = type
        self.keep_dist = keep_dist

    def __enter__(self) -> Self:
        self.value = None
        return super().__enter__()

    def _is_matching(self, msg: Message) -> bool:
        if msg["type"] == self.type:
            for site in self.sites:
                if re.compile(site).fullmatch(msg["name"]) is not None:  # type: ignore[arg-type]
                    return True
        return False

    def _postprocess_message(self, msg: Message) -> None:
        if self.value is None and self._is_matching(msg):
            value = msg["value"]
            assert value is not None
            self.value = value

    def _process_message(self, msg: Message) -> None:
        if self.value is not None and self._is_matching(msg):  # type: ignore[unreachable]
            msg["value"] = self.value  # type: ignore[unreachable]
            if msg["type"] == "sample":
                msg["is_observed"] = True
                if not self.keep_dist:
                    msg["infer"] = {"_deterministic": True}
                    msg["fn"] = Delta(self.value, event_dim=msg["fn"].event_dim).mask(
                        False
                    )
