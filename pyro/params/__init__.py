from __future__ import absolute_import, division, print_function

from .param_store import ParamStoreDict

# the global ParamStore
_PYRO_PARAM_STORE = ParamStoreDict()

# used to create fully-formed param names, e.g. mymodule$$$mysubmodule.weight
_MODULE_NAMESPACE_DIVIDER = "$$$"


def param_with_module_name(pyro_name, param_name):
    return _MODULE_NAMESPACE_DIVIDER.join([pyro_name, param_name])


def module_from_param_with_module_name(param_name):
    return param_name.split(_MODULE_NAMESPACE_DIVIDER)[0]


def user_param_name(param_name):
    if _MODULE_NAMESPACE_DIVIDER in param_name:
        return param_name.split(_MODULE_NAMESPACE_DIVIDER)[1]
    return param_name
