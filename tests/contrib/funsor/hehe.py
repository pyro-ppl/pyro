from funsor.cnf import Contraction
from funsor.tensor import Tensor
import torch
import funsor.ops as ops
from funsor import Bint, Real
from funsor.terms import Unary, Binary, Variable, Number, lazy, to_data
from funsor.constant import Constant
from funsor.delta import Delta
import funsor
funsor.set_backend("torch")

t1 = Unary(ops.exp,
   Contraction(ops.null, ops.add,
    frozenset(),
    (Delta(
      (('x__BOUND_77',
        (Tensor(
          torch.tensor([-1.5227684840130555, 0.38168390241818895, -1.0276085758313882], dtype=torch.float64),  # noqa
          (('plate_outer__BOUND_79',
            Bint[3],),),
          'real'),
         Number(0.0),),),)),
     Constant(
      (('plate_inner__BOUND_78',
        Bint[2],),),
      Tensor(
       torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
       (('plate_outer__BOUND_79',
         Bint[3],),),
       'real')),)))
t2 = Unary(ops.all,
   Binary(ops.eq,
    Tensor(
     torch.tensor([-1.5227684840130555, 0.38168390241818895, -1.0276085758313882], dtype=torch.float64),  # noqa
     (('plate_outer__BOUND_79',
       Bint[3],),),
     'real'),
    Tensor(
     torch.tensor([-1.5227684840130555, 0.38168390241818895, -1.0276085758313882], dtype=torch.float64),  # noqa
     (('plate_outer__BOUND_79',
       Bint[3],),),
     'real')))
t3 = Tensor(
   torch.tensor([[-3.304130052277938, -0.9255234395261538, -1.5122103473560844], [-3.217490519312117, -1.2663745664889694, -1.2900109994655682]], dtype=torch.float64),  # noqa
   (('plate_inner__BOUND_78',
     Bint[2],),
    ('plate_outer__BOUND_79',
     Bint[3],),),
   'real')
t = Contraction(ops.add, ops.mul,
 frozenset({Variable('plate_inner__BOUND_78', Bint[2]), Variable('x__BOUND_77', Real), Variable('plate_outer__BOUND_79', Bint[3])}),  # noqa
 (t1,
  t2,
  t3,))
with lazy:
    term = Contraction(ops.add, ops.mul,
     frozenset({Variable('plate_inner__BOUND_78', Bint[2]), Variable('x__BOUND_77', Real), Variable('plate_outer__BOUND_79', Bint[3])}),  # noqa
     (t1,
      t2,
      t3,))

breakpoint()
x = to_data(funsor.optimizer.apply_optimizer(term))
pass
