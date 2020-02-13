import pyro.poutine.runtime
from .name_messenger import NameMessenger


@pyro.poutine.runtime.effectful(type="to_funsor")
def to_funsor(x, output=None, dim_to_name=None):
    return funsor.to_funsor(x, output=output, dim_to_name=dim_to_name)


@pyro.poutine.runtime.effectful(type="to_data")
def to_data(x, name_to_dim=None):
    return funsor.to_data(x, name_to_dim=name_to_dim)
