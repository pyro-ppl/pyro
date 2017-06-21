class ParamStoreDict(object):

    def __init__(self):
        self._params = {}
        self._tags = {}
        self._param_to_name = {}

    def _clear_cache(self):
        self._params = {}
        self._tags = {}
        self._param_to_name = {}

    def get_param(self, name, init_tensor=None, tag="main"):
        """
        Return named parameters from the global store
        Global store should be adjusted...
        """
        if tag not in self._tags:
            self._tags[tag] = {}

        # get the scoped params
        tag_group = self._tags[tag]

        # make sure the param exists in our group
        if name not in self._params:
            # if not create the init tensor through
            assert init_tensor is not None, "cannot initialize a parameter with None. Did you get the param name right?"

            # a function
            if callable(init_tensor):
                # self._params[name] = pyro.device(init_tensor())
                self._params[name] = init_tensor()
            else:
                # from the memory passed in
                self._params[name] = init_tensor

            # keep track of each tensor and it's name
            self._param_to_name[self._params[name]] = name

        # tag -> params
        if name not in tag_group:
            tag_group[name] = self._params[name]

        # send back the guaranteed to exist param
        return self._params[name]

    def param_name(self, p):
        if p not in self._param_to_name:
            return None

        return self._param_to_name[p]

    # only return parameters matching some tag
    def filter_parameters(self, tag):
        # if you've never seen the tag, return empty
        if tag not in self._tags:
            return []

        return self._tags[tag].values()
