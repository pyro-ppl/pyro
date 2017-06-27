import pyro
import torch.nn as pt_nn
# import inspect


class Object(object):
    pass


nn = Object()

cur_stack = None

# look for all of the objects inside of the nn module
for fct_name in dir(pt_nn):

    nn_mod = getattr(pt_nn, fct_name)

    if not isinstance(nn_mod, dict):
        setattr(nn, fct_name, nn_mod)

    # wrap these objects
    orig_new = getattr(nn_mod, "__new__")

    # capture all new calls, and add to stack when appropriate
    def _new_replace(cls, pyro_name=None, *args, **kwargs):

        # you got initialized, and we sent in a pyro_name object
        # build a current stack
        if pyro_name is not None:
            global cur_stack
            prev_stack = cur_stack
            cur_stack = []
            # pyro_name is the unique id
            setattr(nn, pyro_name, 0)

            # temporarily overload
            # time to do shady shit
            PT_Paramter = pt_nn.Parameter

            # aha! time to do some shady shit
            def pt_param_call(*args, **kwargs):
                uid = getattr(nn, pyro_name)
                unique_param_name = "_".join(cur_stack) + str(uid)
                setattr(nn, pyro_name, uid + 1)
                rv = PT_Paramter(*args, **kwargs)
                return pyro.param(unique_param_name, rv, group=pyro_name)

            # overwrite current with overloaded fct
            setattr(pt_nn, "Parameter", pt_param_call)

        # don't bother, nothing is happening
        if cur_stack is None:
            return orig_new(*args, **kwargs)

        # add to stack
        cur_stack.append(cls.__name__)

        # if any calls to pt param are made, they'll be caught in above fct
        rv = orig_new(cls, *args, **kwargs)

        # no more of this new fct
        cur_stack.pop()

        if pyro_name is not None:
            cur_stack = prev_stack
            # set back to original
            setattr(pt_nn, "Parameter", PT_Paramter)

        return rv

    # def fct_replace(pyro_name=None, *args, **kwargs):
#   if pyro_name != None:
#     # we need to do something
#     # here we scope/wrap all param calls with named pyro.param
#     # calls

#   # all done setting
#   lm = module_class_constructor(*args, **kwargs)

#   # clear state of whatever we did above
#   if pyro_name != None:
#     # clear

#   return lm

    # setattr(nn, )

# module_class_constructor = getattr(torch.nn, "Module")

# # going to replace this module construction object
# # e.g. fct_name = Linear or fct_name = LSTM
# def fct_replace(pyro_name=None, *args, **kwargs):
#   if pyro_name != None:
#     # we need to do something
#     # here we scope/wrap all param calls with named pyro.param
#     # calls

#   # all done setting
#   lm = module_class_constructor(*args, **kwargs)

#   # clear state of whatever we did above
#   if pyro_name != None:
#     # clear

#   return lm


# setattr(torch.nn, "Module")


# # look for all of the objects inside of the nn module
# for fct_name in dir(nn):

#   # if we're not a subclass of nn.Module, ignore this object
#   # old: if not inspect.isclass(getattr(nn, fct_name)):
#   if not issubclass(getattr(nn, fct_name), nn.Module):
#     continue

#   class_constructor = getattr(nn, fct_name)


#   #
#   setattr(nn, fct_name, fct_replace)


# # later on
# import pyro.nn as nn


# nn.Linear(pyro_name="dope")
