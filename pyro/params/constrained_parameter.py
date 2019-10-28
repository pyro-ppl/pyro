import weakref

import torch
from torch.distributions import transform_to
from torch.distributions.constraints import Constraint


class ConstrainedParameter:
    """
    Constrained wrapper around a :class:`~torch.nn.Parameter` that obeys a
    :class:`~torch.distributions.constraints.Constraint` .

    Like :class:`~torch.nn.Parameter` , this can be accessed directly as an
    attribute of an enclosing :class:`~torch.nn.Module` . Unlike a
    :class:`~torch.nn.Parameter` , the ``.data`` attribute cannot be set
    directly; instead set data via the :meth:`unconstrained` method::

        my_module.scale = ConstrainedParameter(torch.ones(2,3,4),
                                               constraint=constraints.positive)
        assert isinstance(my_module.scale, torch.Tensor)
        my_module.scale.unconstrained().data.normal_()

    ConstrainedParameters can be owned by only one object.

    :param torch.Tensor constrained_data: Initial data in constrained space.
    :param ~torch.distributions.constraints.Constraint constraint: A
        constraint.
    """

    def __init__(self, constrained_data, constraint):
        assert isinstance(constrained_data, torch.Tensor)
        assert isinstance(constraint, Constraint)
        self.constraint = constraint
        self._owner = None
        self._name = None

        super().__init__()
        self.set(constrained_data)

    def unconstrained(self):
        """
        Provides access to the underlying unconstrained data.

        :rtype: torch.Tensor
        """
        if self._owner is not None:
            self._unconstrained_value = getattr(self._owner, self._name)
        return self._unconstrained_value

    def get(self):
        """
        Gets the current constrained value.

        :rtype: torch.Tensor
        """
        return self.__get__(None)

    def set(self, constrained_data):
        """
        Sets a new constrained value.

        :param torch.Tensor constrained_data: A new constrained value.
        """
        with torch.no_grad():
            constrained_data = constrained_data.detach()
            unconstrained_data = transform_to(self.constraint).inv(constrained_data)
            unconstrained_data = unconstrained_data.contiguous()
        self._unconstrained_value = torch.nn.Parameter(unconstrained_data)
        if self._owner is not None:
            setattr(owner, self._name, self._unconstrained_value)

    def _update_owner(self, owner):
        if owner is not self._owner:
            self._owner = owner
            for name, attr in owner.__dict__.items():
                if attr is self:
                    self._name = name + "_unconstrained"
                    return
        raise ValueError("ConstrainedParameter is not owned by {}".format(owner))

    def __get__(self, owner, obj_type=None):
        if owner is not None:
            self._update_owner(owner)
        unconstrained_value = self.unconstrained()
        constrained_value = transform_to(self.constraint)(unconstrained_value)

        # We add a weakref to the constrained result to provide uniform access
        # to the underlying data:
        # 1. When called on a free-standing instance, .unconstrained() will
        #    call the above unconstrained() method.
        # 2. When called on an attribute of an enclosing object, that object's
        #    getattr will trigger, __get__, returning a constrained
        #    torch.Tensor (not a ConsterainedParameter), and .unconstrained()
        #    will dereference the following weakref:
        constrained_value.unconstrained = weakref.ref(unconstrained_value)

        return constrained_value

    # These must be defined to ensure this is a data attribute,
    # but cannot be implemented because torch.nn.Module implements
    # custom .__setattr__() and .__delattr__() methods.
    def __set__(self, owner, constrained_data):
        raise AttributeError('Not Implemented')

    def __delete__(self, owner):
        raise AttributeError('Not Implemented')
