import pyro

# wrap the optim object


class PyroOptim(object):

    # helper to fetch the optim args if callable
    def get_optim_args(self, param):
        # if we were passed a fct, we call fct with param info
        # arguments are (module name, param name) e.g. ('mymodule', 'bias')
        if callable(self.pt_optim_args):

            # get param name
            param_name = pyro._param_store.param_name(param)
            mod_name = pyro.module_from_param_name(param_name)
            stripped_param_name = pyro.user_param_name(param_name)
            opt_dict = self.pt_optim_args(mod_name, stripped_param_name)

            # must be dictionary
            assert isinstance(
                opt_dict, dict), "per-param optim arg must return defaults dictionary"
            return opt_dict
        else:
            return self.pt_optim_args

    # initialize with a particular sgd
    def __init__(self, optim_constructor, optim_args):

        # will call the constructor for the optim function for every new param
        # spotted
        self.pt_optim_constructor = optim_constructor

        # must be callable or tuple
        assert callable(optim_args) or isinstance(
            optim_args, dict), "optim_args must be function that returns defaults or a defaults dictionary"

        # hold our args to be called/used
        self.pt_optim_args = optim_args
        self.optim_objs = {}

    # when called, check params for
    def __call__(self, params, closure=None, *args, **kwargs):

        # if you have closure, according to optim, you calc the loss
        # TODO: Warning, this supports mostly all normal closure behavior, e.g. adam, sgd, asgd
        # but not necessarily for cases where the closure is called more than
        # once by the optim step fct
        loss = None if closure is None else closure()

        # we collect all relevant optim objects
        active_optims = []

        # loop over relevant params
        for p in params:

            # if we have not seen this param before, we instantiate and optim
            # obj to deal with it
            if p not in self.optim_objs:

                # get our constructor arguments
                def_optim_dict = self.get_optim_args(p)

                # create a single optim object for that param
                self.optim_objs[p] = self.pt_optim_constructor(
                    [p], **def_optim_dict)

            # we add this to the list of params we'll be stepping on
            active_optims.append(self.optim_objs[p])

        # actually perform the step for each optim obj
        for ix, opt_obj in enumerate(active_optims):
            opt_obj.step(*args, **kwargs)

        # all done, send back loss if calculated
        return loss
