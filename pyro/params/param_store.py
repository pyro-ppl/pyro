from collections import defaultdict
import cloudpickle


class ParamStoreDict(object):

    def __init__(self):
        self._params = {}
        self._param_to_name = {}
        self._active_params = set()
        self._param_scopes = defaultdict(lambda: set())

    def clear(self):
        """
        clear the parameter store
        """
        self._params = {}
        self._param_to_name = {}
        self._active_params = set()
        self._param_scopes = defaultdict(lambda: set())

    def get_all_param_names(self):
        """
        get all parameter names in param store
        """
        return self._params.keys()

    def get_active_params(self, scope=None):
        """
        :param scope: optional argument specifying that only active params of a particular
            scope or scopes should be returned
        :type scope: string or iterable over strings
        :returns: all active params in the ParamStore, possibly filtered to a particular scope or scopes
        :rtype: set
        """
        if scope is None:  # return all active params
            return self._active_params
        elif isinstance(scope, str) and scope not in self._param_scopes:
            # return empty set, since scope doesn't exist; XXX RAISE WARNING
            return set()
        elif isinstance(scope, str):  # only return active params in the scope
            return self._active_params.intersection(self._param_scopes[scope])
        elif isinstance(scope, list) or isinstance(scope, tuple):
            params_to_return = set()
            for s in scope:
                assert isinstance(s, str)
                if s in self._param_scopes:
                    params_to_return.update(self._param_scopes[s])
            return params_to_return.intersection(self._active_params)
        else:
            raise TypeError

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

    def delete_scope(self, scope):
        """
        :param scope: scope to remove
        :type socpe: str

        Removes the scope; any parameters in that scope are untouched but are no longer
        associated with that scope.
        """
        self._param_scopes.pop(scope)

    def add_param_to_scope(self, param_name, scope):
        """
        :param param_name: either a single parameter name or an iterable of parameter names
        :param scope: either a single string or an iterable of strings

        Adds the parameter(s) specified by param_name to the scope(s) specified by scope.
        """

        def add_single_param_to_scope(name, scope):
            assert name in self._params
            if isinstance(scope, str):
                self._param_scopes[scope].add(self._params[name])
            else:
                for s in scope:
                    assert isinstance(s, str), "scope must be a string or an iterable of strings"
                    self._param_scopes[s].add(self._params[name])

        if isinstance(param_name, str):
            add_single_param_to_scope(param_name, scope)
        else:
            for p in param_name:
                assert isinstance(p, str), "param_name must be a string or an iterable of strings"
                add_single_param_to_scope(p, scope)

    def remove_param_from_scope(self, param_name, scope):
        """
        :param param_name: either a single parameter name or an iterable of parameter names
        :param scope: either a single string or an iterable of strings

        Removes the parameter(s) specified by param_name to the scope(s) specified by scope.
        The parameter(s) are unchanged but will no longer be associated with the specified scope(s).
        """

        def remove_single_param_from_scope(name, scope):
            assert name in self._params
            if isinstance(scope, str):
                self._param_scopes[scope].discard(self._params[name])
            else:
                for s in scope:
                    assert isinstance(s, str), "scope must be a string or an iterable of strings"
                    self._param_scopes[s].discard(self._params[name])

        if isinstance(param_name, str):
            remove_single_param_from_scope(param_name, scope)
        else:
            for p in param_name:
                assert isinstance(p, str), "param_name must be a string or an iterable of strings"
                remove_single_param_from_scope(p, scope)

    def get_param(self, name, init_tensor=None, scope="default"):
        """
        :param name: parameter name
        :type name: str
        :param init_tensor: initial tensor
        :type init_tensor: torch.autograd.Variable
        :param scope: the scope to assign to the parameter
        :type scope: a string or iterable of strings
        :returns: parameter
        :rtype: torch.autograd.Variable

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
            self.add_param_to_scope(name, scope)

        # send back the guaranteed to exist param
        return self._params[name]

    def param_name(self, p):
        """
        XXX Is this used anywhere???
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

        Save parameters to disk XXX FIX ME (include scopes etc.)
        """
        with open(filename, "wb") as output_file:
            output_file.write(cloudpickle.dumps(self._params))

    def load(self, filename):
        """
        :param filename: file name to load from
        :type name: str

        Loads parameters from disk XXX FIX ME (include scopes etc.)
        """
        with open(filename, "rb") as input_file:
            loaded_params = cloudpickle.loads(input_file.read())
            for param_name, param in loaded_params.items():
                self._params[param_name] = param
                self._param_to_name[param] = param_name
