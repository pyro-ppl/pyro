from collections import defaultdict
import cloudpickle


class ParamStoreDict(object):
    """
    Global store for parameters in pyro. The typical user interacts with the paramstore
    primarily through the primitive pyro.param.
    """

    def __init__(self):
        """
        initialize param store data structures
        """
        self._params = {}  # dictionary from param name to param
        self._param_to_name = {}  # dictionary from param to param name
        self._active_params = set()  # set of all currently active params
        self._param_tags = defaultdict(lambda: set())  # dictionary from tag to param names
        self._tag_params = defaultdict(lambda: set())  # dictionary from param name to tags

    def clear(self):
        """
        clear the parameter store
        """
        self._params = {}
        self._param_to_name = {}
        self._active_params = set()
        self._param_tags = defaultdict(lambda: set())
        self._tag_params = defaultdict(lambda: set())

    def get_all_param_names(self):
        """
        get all parameter names in param store
        """
        return self._params.keys()

    def get_active_params(self, tags=None):
        """
        :param tag: optional argument specifying that only active params carrying a particular
            tag or any of several tags should be returned
        :type tags: string or iterable over strings
        :returns: all active params in the ParamStore, possibly filtered to a particular tag or tags
        :rtype: set
        """
        if tags is None:  # return all active params
            return self._active_params
        elif isinstance(tags, str) and tags not in self._param_tags:
            # return empty set, since tag doesn't exist; XXXraise warning?
            return set()
        elif isinstance(tags, str):  # only return active params in the tag
            return self._active_params.intersection(self._param_tags[tags])
        elif isinstance(tags, list) or isinstance(tags, tuple):
            params_to_return = set()
            for tag in tags:
                assert isinstance(tag, str)
                if tag in self._param_tags:
                    params_to_return.update(self._param_tags[tag])
            return params_to_return.intersection(self._active_params)
        else:
            raise TypeError

    def mark_params_active(self, params):
        """
        :param params: iterable of params the user wishes to mark as active in the ParamStore.
            this information is used to determine which parameters are being optimized,
            e.g. in the context of pyro.infer.SVI
        """
        assert(all([p in self._param_to_name for p in params])), \
            "some of these parameters are not in the param store"
        self._active_params.update(set(params))

    def mark_params_inactive(self, params):
        """
        :param params: iterable of params the user wishes to mark as inactive in the ParamStore.
            this information is used to determine which parameters are being optimized,
            e.g. in the context of pyro.infer.SVI
        """
        assert(all([p in self._param_to_name for p in params])), \
            "some of these parameters are not in the param store"
        self._active_params.difference_update(set(params))

    def delete_tag(self, tag):
        """
        :param tag: tag to remove
        :type tag: str

        Removes the tag; any parameters with that tag are unaffected but are no longer
        associated with that tag.
        """
        assert(tag in self._param_tags), "this tag does not exist"
        self._param_tags.pop(tag)
        for p, tags in self._tag_params.items():
            if tag in tags:
                tags.remove(tag)

    def get_param_tags(self, param_name):
        """
        :param param_name: a (single) parameter name
        :type param_name: str
        :rtype: set

        Return the tags associated with the parameter
        """
        if param_name in self._tag_params:
            return self._tag_params[param_name]
        return set()

    def tag_params(self, param_names, tags):
        """
        :param param_name: either a single parameter name or an iterable of parameter names
        :param tags: either a single string or an iterable of strings

        Tags the parameter(s) specified by param_names with the tag(s) specified by tags.
        """
        def tag_single_param(name, tags):
            assert name in self._params, "<%s> is not a parameter in the paramstore" % name
            if isinstance(tags, str):
                self._param_tags[tags].add(self._params[name])
                self._tag_params[name].add(tags)
            else:
                for tag in tags:
                    assert isinstance(tag, str), "tags must be a string or an iterable of strings"
                    self._param_tags[tag].add(self._params[name])
                    self._tag_params[name].add(tag)

        if isinstance(param_names, str):
            tag_single_param(param_names, tags)
        else:
            for p in param_names:
                assert isinstance(p, str), "param_names must be a string or an iterable of strings"
                tag_single_param(p, tags)

    def untag_params(self, param_names, tags):
        """
        :param param_name: either a single parameter name or an iterable of parameter names
        :param tags: either a single string or an iterable of strings

        Disassociates the parameter(s) specified by param_names with the tag(s) specified by tags.
        """
        def untag_single_param(name, tags):
            assert name in self._params, "<%s> is not a parameter in the paramstore" % name
            if isinstance(tags, str):
                self._param_tags[tags].discard(self._params[name])
                self._tag_params[name].discard(tags)
            else:
                for tag in tags:
                    assert isinstance(tag, str), "tags must be a string or an iterable of strings"
                    self._param_tags[tag].discard(self._params[name])
                    self._tag_params[name].discard(tag)

        if isinstance(param_names, str):
            untag_single_param(param_names, tags)
        else:
            for p in param_names:
                assert isinstance(p, str), "param_names must be a string or an iterable of strings"
                untag_single_param(p, tags)

    def get_param(self, name, init_tensor=None, tags="default"):
        """
        :param name: parameter name
        :type name: str
        :param init_tensor: initial tensor
        :type init_tensor: torch.autograd.Variable
        :param tags: the tag(s) to assign to the parameter
        :type tags: a string or iterable of strings
        :returns: parameter
        :rtype: torch.autograd.Variable

        Get parameter from its name. If it does not yet exist in the
        param store, it will be created and stored
        """
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

            # keep track of param tags
            self.tag_params(name, tags)

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

        Save parameters to disk
        """
        with open(filename, "wb") as output_file:
            param_tags = {k: [tag for tag in self._param_tags[k]] for k in self._param_tags}
            to_save = (self._params, param_tags)
            output_file.write(cloudpickle.dumps(to_save))

    def load(self, filename):
        """
        :param filename: file name to load from
        :type name: str

        Loads parameters from disk
        """
        with open(filename, "rb") as input_file:
            loaded_param_data = cloudpickle.loads(input_file.read())
            loaded_params, loaded_param_tags = loaded_param_data

            for param_name, param in loaded_params.items():
                self._params[param_name] = param
                self._param_to_name[param] = param_name

            for param_name, tags in loaded_param_tags.items():
                for tag in tags:
                    self._param_tags[param_name].add(tag)
                    self._tag_params[tag].add(param)
