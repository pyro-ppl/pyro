from collections import OrderedDict, namedtuple

import torch
from torch.distributions import transform_to

ConstrainedParameter = namedtuple("ConstrainedParameter", ("data", "constraint"))


class ConstrainedModule(torch.nn.Module):
    """
    Subclass of :class:`~torch.nn.Module` that allows dynamically defined
    :class:`~torch.distributions.constraints.Constraint` on its
    :class:`~torch.nn.Parameter`. To declare a constrained parameter, use the
    :class:`ConstrainedParameter` helper::

        my_module = ConstrainedModule()
        my_module.x = ConstrainedParameter(torch.tensor(1.),
                                           constraints.positive)

        # Now scale is a function of an unconstrained nn.Parameter.
        assert isinstance(my_module.x, torch.Tensor)
        assert isinstance(my_module.x_unconstrained, nn.Parameter)

    Like an unconstrained :class:`~torch.nn.Parameter` ,
    a :class:`ConstrainedParameter` can be accessed directly as an attribute of
    its :class:`ConstrainedModule` . Unlike an unconstrained
    :class:`~torch.nn.Parameter` , the ``.data`` attribute cannot be set
    directly; instead set data of the correspondingly named parameter appended
    with the string "_unconstrained"::

        # Correct way to initialze:
        my_module.x_unconstrained.data.normal_()

        # XXX Wrong way to initialize XXX
        # my_module.x.data.normal_()  # has no effect.
    """
    def __init__(self):
        self._constraints = OrderedDict()
        super().__init__()

    def __setattr__(self, name, value):
        if '_constraints' not in self.__dict__:
            return super().__setattr__(name, value)
        _constraints = self.__dict__['_constraints']

        if isinstance(value, ConstrainedParameter):
            constrained_value, constraint = value
        elif name in _constraints:
            constrained_value = value
            constraint = _constraints[name]
        else:
            return super().__setattr__(name, value)

        self._constraints[name] = constraint
        with torch.no_grad():
            constrained_value = constrained_value.detach()
            unconstrained_value = transform_to(constraint).inv(constrained_value)
            unconstrained_value = unconstrained_value.contiguous()
        unconstrained_value = torch.nn.Parameter(unconstrained_value)
        setattr(self, name + "_unconstrained", unconstrained_value)

    def __getattr__(self, name):
        if '_constraints' not in self.__dict__:
            return super().__getattr__(name)
        _constraints = self.__dict__['_constraints']

        if name not in _constraints:
            return super().__getattr__(name)

        constraint = _constraints[name]
        unconstrained_value = getattr(self, name + "_unconstrained")
        return transform_to(constraint)(unconstrained_value)

    def __delattr__(self, name):
        if '_constraints' not in self.__dict__:
            return super().__delattr__(name)
        _constraints = self.__dict__['_constraints']

        if name not in _constraints:
            return super().__delattr__(name)

        delattr(self, name + "_unconstrained")
        del _constraints[name]

    def constraints(self):
        """
        Returns an iterator over parameter (name,constraint) pairs.

        :yields: (str, Constraint) Tuple containing the name and constraint.
        """
        for name, constraint in self._constraints.items():
            yield name, constraint


class ConstraintDescriptor:
    """
    Descriptor to add a static
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
    Decorator for static constrained parameters. For example::

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
    return ConstraintDescriptor(name, constraint_fn)
