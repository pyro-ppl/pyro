import pyro.nn as nn


rr = nn.Linear(2, 2)


r2 = nn.Linear(2, 2, pyro_name="dope")

# import pyro
# import torch.nn as nn

# cur_stack = []

# # extend the nn module from pytorch
# class Module(nn.Module):

#   def __init__(self, pyro_name=None, *args, **kwargs):
#     # if no pyro name, ignore this call
#     if pyro_name != None:
#       super(Module, self).__init__(*args, **kwargs)
#     else:
#       uid = 0

#       #
#       PT_Paramter = nn.Parameter
#       nm_new_original = nn.Module.__new__

#       # anytime a module is constructed
#       # catch that sucka
#       def new_overload(cls, *args, **kwargsx):
#         cur_stack.append(cls.__name__)
#         rv = nm_new_original(cls, *args, **kwargs)
#         cur_stack.pop()
#         return rv

#       nn.Module.__new__ = new_overload

#       # aha! time to do some shady shit
#       def pt_param_call(*args, **kwargs):
#         unique_param_name = "_".join(cur_stack) + str(uid)
#         uid += 1
#         rv = PT_Paramter(*args, **kwargs)
#         return pyro.param(unique_param_name, rv, group=pyro_name)

#       # now we're done
#       cur_stack = []

#       # replace as if nothing happened
#       nn.Parameter = PT_Paramter
#       nn.Module.__new__ = nm_new_original
