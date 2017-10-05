from collections import defaultdict
import cloudpickle


class ParamStoreDict(object):

    def __init__(self):
        self._params = {}
        self._param_to_name = {}
        self._active_params = set()
        self._param_scopes = defaultdict(lambda: set())

    def clear(self):
        self._params = {}
        self._param_to_name = {}
        self._active_params = set()
        self._param_scopes = defaultdict(lambda: set())

    def get_active_params(self, scope=None):
        """
        returns all active params in the ParamStore as a set.
        """
        if scope is None:
            return self._active_params
        else:
            return self._active_params.intersection(self._param_scopes[scope])

    def mark_params_active(self, params):
        """
        :param params: iterable of params the user wishes to mark as active in the ParamStore.
            this information is used by pyro.optim.Optimize
        """
        self._active_params.update(set(params))

    def mark_params_inactive(self, params):
        """
        :param params: iterable of params the user wishes to mark as inactive in the ParamStore.
            this information is used by pyro.optim.Optimize
        """
        self._active_params.difference_update(set(params))

    def get_param(self, name, init_tensor=None, scope="default"):
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

            # keep track of param scope
            if isinstance(scope, str):
                self._param_scopes[scope].add(self._params[name])
            else:
                for s in scope:
                    assert isinstance(s, str), "scope must be a string or an iterable of strings"
                    self._param_scopes[s].add(self._params[name])

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
