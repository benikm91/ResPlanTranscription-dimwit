package resplan.nn

import dimwit.*

object Util:
  def vmap[T <: Tuple: Labels, T2 <: Tuple: Labels, L: Label, V1, V2](axis: Axis[L])(f: (Tensor[T, V1] => Tensor[T2, V2])): Tensor[L *: T, V1] => Tensor[L *: T2, V2] =
    x => x.vmap(axis)(f)
