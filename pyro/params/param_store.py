import cloudpickle


class ParamStoreDict(object):

    def __init__(self):
        self._params = {}
        self._param_to_name = {}

    def clear(self):
        self._params = {}
        self._param_to_name = {}

    #def set_param_active()
    #def set_param_inactive()


    def get_param(self, name, init_tensor=None):
        """
        :param name: parameter name
        :type name: str
        :param init_tensor: initial tensor
        :returns: parameter

        Get parameter from its name. If it does not yet exist in the
        param store, it will be created and stored
        """
        # make sure the param exists in our group
        if name not in self._params:
            # if not create the init tensor through
            assert init_tensor is not None,\
                "cannot initialize a parameter with None. Did you get the param name right?"

            # a function
            if callable(init_tensor):
                self._params[name] = init_tensor()
            else:
                # from the memory passed in
                self._params[name] = init_tensor

            # keep track of each tensor and it's name
            self._param_to_name[self._params[name]] = name

        # send back the guaranteed to exist param
        return self._params[name]

    def param_name(self, p):
        """
        :param p: parameter
        :returns: parameter name

        Get parameter name from parameter
        """
        if p not in self._param_to_name:
            return None

        return self._param_to_name[p]

    def save(self, filename):
        """
        :param filename: file name to save to
        :type name: str

        Save parameters to disk
        """
        with open(filename, "wb") as output_file:
            output_file.write(cloudpickle.dumps(self._params))

    def load(self, filename):
        """
        :param filename: file name to load from
        :type name: str

        Loads parameters from disk
        """
        with open(filename, "rb") as input_file:
            loaded_params = cloudpickle.loads(input_file.read())
            for param_name, param in loaded_params.items():
                self._params[param_name] = param
                self._param_to_name[param] = param_name
