package resplan.nn.normalization

import dimwit.*
import dimwit.Conversions.given

case class RMSNorm[L: Label](params: RMSNorm.Params[L]) extends (Tensor1[L, Float] => Tensor1[L, Float]):

  private def rescale(x: Tensor1[L, Float]): Tensor1[L, Float] =
    val variance = (x -! x.mean).pow(2).mean
    val epsilon = 1e-6f
    x /! (variance + epsilon).sqrt

  def apply(x: Tensor1[L, Float]): Tensor1[L, Float] =
    rescale(x) * params.weight

object RMSNorm:

  case class Params[L](weight: Tensor1[L, Float])

  object Params:
    def defaultInit[L: Label](ae: AxisExtent[L]) =
      Params(weight = Tensor(Shape(ae)).fill(1f))
