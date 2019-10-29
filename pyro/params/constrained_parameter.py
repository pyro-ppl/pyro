import torch
from torch.distributions import transform_to


class ConstrainedParameter:
    """
    Descriptor to add a
    :class:`~torch.distributions.constraints.Constraint` to a
    :class:`~torch.nn.Parameter` of a :class:`~torch.nn.Module` .
    These are typically created via the :func:`constraint` decorator.

    Like :class:`~torch.nn.Parameter` , these can be accessed directly as an
    attribute of an enclosing :class:`~torch.nn.Module` . Unlike a
    :class:`~torch.nn.Parameter` , the ``.data`` attribute cannot be set
    directly; instead set data of the correspondingly named parameter appended
    with the string "_unconstrained"::

        class MyModule(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            @constraint
            def x(self):
                return contraints.positive

        my_module = MyModule(torch.randn(3,4))

        # Correct way to initialze:
        my_module.x_unconstrained.data.normal_()

        # XXX Wrong way to initialize XXX
        # my_module.x.data.normal_()  # has no effect.

    Constraints may depend on other parameters (of course not cyclically), for
    example::

        class Interval(nn.Module):
            def __init__(self, lb, ub):
                super().__init__()
                self.lb = lb
                self.ub = ub

            @constraint
            def ub(self):
                return constraints.greater_than(self.lb)

    :param str name: The name of the constrained parameter.
    :param callable constraint: A function that inputs a
        :class:`~torch.nn.Module` and returns a
        :class:`~torch.distributions.constraints.Constraint` object.
    """

    def __init__(self, name, constraint_fn):
        assert isinstance(name, str)
        assert callable(constraint_fn)
        self.name = name
        self._unconstrained_name = name + "_unconstrained"
        self._constraint_fn = constraint_fn

    def __get__(self, obj, obj_type=None):
        if obj is None:
            return self

        constraint = self._constraint_fn(obj)
        unconstrained_value = getattr(obj, self._unconstrained_name)
        constrained_value = transform_to(constraint)(unconstrained_value)
        return constrained_value

    def __set__(self, obj, constrained_value):
        with torch.no_grad():
            constraint = self._constraint_fn(obj)
            constrained_value = constrained_value.detach()
            unconstrained_value = transform_to(constraint).inv(constrained_value)
            unconstrained_value = unconstrained_value.contiguous()
        setattr(obj, self._unconstrained_name, torch.nn.Parameter(unconstrained_value))

    def __delete__(self, obj):
        delattr(obj, self._unconstrained_name)


def constraint(constraint_fn):
    """
    Decorator for constrained parameters. For example::

        class Normal(nn.Module):
            def __init__(self, loc, scale):
                super().__init__()
                self.loc = loc
                self.scale = scale

            @constraint
            def scale(self):
                return constraints.positive
    """
    assert callable(constraint_fn)
    name = constraint_fn.__name__
    return ConstrainedParameter(name, constraint_fn)
