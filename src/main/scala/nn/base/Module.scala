package resplan.nn.base

trait Module[R, In, Out] extends (In => R ?=> Out):
  type Eff[T] = R ?=> T

  def apply(i: In): Eff[Out]

trait Module2[R, In1, In2, Out] extends ((In1, In2) => R ?=> Out):
  type Eff[T] = R ?=> T

  def apply(i1: In1, i2: In2): Eff[Out]
