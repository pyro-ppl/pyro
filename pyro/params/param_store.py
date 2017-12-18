from __future__ import absolute_import, division, print_function

from collections import defaultdict

import cloudpickle


class ParamStoreDict(object):
    """
    Global store for parameters in Pyro. This is basically a key-value store.
    The typical user interacts with the ParamStore primarily through the
    primitive `pyro.param`.

    See `Intro Part II <http://pyro.ai/examples/intro_part_ii.html>`_ for further discussion
    and `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for some examples.

    Some things to bear in mind when using parameters in Pyro:

    - parameters must be assigned unique names
    - the `init_tensor` argument to `pyro.param` is only used the first time that a given (named)
      parameter is registered with Pyro.
    - for this reason, a user may need to use the `clear()` method if working in a REPL in order to
      get the desired behavior. this method can also be invoked with `pyro.clear_param_store()`.
    - the internal name of a parameter within a PyTorch `nn.Module` that has been registered with
      Pyro is prepended with the Pyro name of the module. so nothing prevents the user from having
      two different modules each of which contains a parameter named `weight`. by contrast, a user
      can only have one top-level parameter named `weight` (outside of any module).
    - parameters can be 'tagged' with (string) tags. by default each parameter is tagged with the
      'default' tag. this mechanism allows the user to group parameters together and e.g. customize
      learning rates for different tags. for an example where this is useful see the tutorial
      `SVI Part III <http://pyro.ai/examples/svi_part_iii.html>`_.
    - parameters can be saved and loaded from disk using `save` and `load`.
    """

    def __init__(self):
        """
        initialize ParamStore data structures
        """
        self._params = {}  # dictionary from param name to param
        self._param_to_name = {}  # dictionary from param to param name
        self._active_params = set()  # set of all currently active params
        self._param_tags = defaultdict(lambda: set())  # dictionary from tag to param names
        self._tag_params = defaultdict(lambda: set())  # dictionary from param name to tags

    def clear(self):
        """
        Clear the ParamStore
        """
        self._params = {}
        self._param_to_name = {}
        self._active_params = set()
        self._param_tags = defaultdict(lambda: set())
        self._tag_params = defaultdict(lambda: set())

    def named_parameters(self):
        """
        Returns an iterator over tuples of the form (name, parameter) for each parameter in the ParamStore
        """
        return self._params.items()

    def get_all_param_names(self):
        """
        Get all parameter names in the ParamStore
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
            # return empty set, since tag doesn't exist; XXX raise warning?
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
            "some of these parameters are not in the ParamStore"
        self._active_params.update(set(params))

    def mark_params_inactive(self, params):
        """
        :param params: iterable of params the user wishes to mark as inactive in the ParamStore.
            this information is used to determine which parameters are being optimized,
            e.g. in the context of pyro.infer.SVI
        """
        assert(all([p in self._param_to_name for p in params])), \
            "some of these parameters are not in the ParamStore"
        self._active_params.difference_update(set(params))

    def delete_tag(self, tag):
        """
        Removes the tag; any parameters with that tag are unaffected but are no longer
        associated with that tag.

        :param tag: tag to remove
        :type tag: str
        """
        assert(tag in self._param_tags), "this tag does not exist"
        self._param_tags.pop(tag)
        for p, tags in self._tag_params.items():
            if tag in tags:
                tags.remove(tag)

    def get_param_tags(self, param_name):
        """
        Return the tags associated with the parameter

        :param param_name: a (single) parameter name
        :type param_name: str
        :rtype: set
        """
        if param_name in self._tag_params:
            return self._tag_params[param_name]
        return set()

    def tag_params(self, param_names, tags):
        """
        Tags the parameter(s) specified by param_names with the tag(s) specified by tags.

        :param param_name: either a single parameter name or an iterable of parameter names
        :param tags: either a single string or an iterable of strings
        """
        def tag_single_param(name, tags):
            assert name in self._params, "<%s> is not a parameter in the ParamStore" % name
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
        Disassociates the parameter(s) specified by param_names with the tag(s) specified by tags.

        :param param_name: either a single parameter name or an iterable of parameter names
        :param tags: either a single string or an iterable of strings
        """
        def untag_single_param(name, tags):
            assert name in self._params, "<%s> is not a parameter in the ParamStore" % name
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

    def replace_param(self, param_name, new_param, old_param):
        """
        Replace the param param_name with current value old_param with the new value new_param

        :param param_name: parameter name
        :type param_name: str
        :param new_param: the paramater to be put into the ParamStore
        :type new_param: torch.autograd.Variable
        :param old_param: the paramater to be removed from the ParamStore
        :type new_param: torch.autograd.Variable
        """
        assert id(self._params[param_name]) == id(old_param)
        self._params[param_name] = new_param
        self._param_to_name[new_param] = param_name
        self._param_to_name.pop(old_param)

    def get_param(self, name, init_tensor=None, tags="default"):
        """
        Get parameter from its name. If it does not yet exist in the
        ParamStore, it will be created and stored.
        The Pyro primitive `pyro.param` dispatches to this method.

        :param name: parameter name
        :type name: str
        :param init_tensor: initial tensor
        :type init_tensor: torch.autograd.Variable
        :param tags: the tag(s) to assign to the parameter
        :type tags: a string or iterable of strings
        :returns: parameter
        :rtype: torch.autograd.Variable
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
        Get parameter name from parameter

        :param p: parameter
        :returns: parameter name
        """
        if p not in self._param_to_name:
            return None

        return self._param_to_name[p]

    def get_state(self):
        """
        Get the ParamStore state.
        """
        param_tags = {k: list(tags) for k, tags in self._param_tags.items()}
        state = (self._params, param_tags)
        return state

    def set_state(self, state):
        """
        Set the ParamStore state using state from a previous get_state() call
        """
        assert isinstance(state, tuple) and len(state) == 2, "malformed ParamStore state"
        loaded_params, loaded_param_tags = state

        for param_name, param in loaded_params.items():
            self._params[param_name] = param
            self._param_to_name[param] = param_name

        for param_name, tags in loaded_param_tags.items():
            for tag in tags:
                self._param_tags[param_name].add(tag)

    def save(self, filename):
        """
        Save parameters to disk

        :param filename: file name to save to
        :type name: str
        """
        with open(filename, "wb") as output_file:
            output_file.write(cloudpickle.dumps(self.get_state()))

    def load(self, filename):
        """
        Loads parameters from disk

        :param filename: file name to load from
        :type name: str
        """
        with open(filename, "rb") as input_file:
            state = cloudpickle.loads(input_file.read())
        self.set_state(state)
