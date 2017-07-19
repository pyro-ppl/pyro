import pyro


def param_with_module_name(pyro_name, param_name):
    return pyro._MODULE_NAMESPACE_DIVIDER.join([pyro_name, param_name])


def module_from_param_with_module_name(param_name):
    return param_name.split(pyro._MODULE_NAMESPACE_DIVIDER)[0]


def user_param_name(param_name):
    if pyro._MODULE_NAMESPACE_DIVIDER in param_name:
        return param_name.split(pyro._MODULE_NAMESPACE_DIVIDER)[1]
    return param_name
