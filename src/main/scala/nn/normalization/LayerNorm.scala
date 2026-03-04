package resplan.nn.normalization

import dimwit.*
import dimwit.Conversions.given

case class LayerNorm[L: Label](params: LayerNorm.Params[L]) extends (Tensor1[L, Float] => Tensor1[L, Float]):

  private def standardize(x: Tensor1[L, Float]): Tensor1[L, Float] =
    val x0 = x -! x.mean
    val variance = x0.pow(2).mean
    val epsilon = 1e-6f
    x0 /! (variance + epsilon).sqrt

  def apply(x: Tensor1[L, Float]): Tensor1[L, Float] =
    standardize(x) * params.weight + params.bias

object LayerNorm:

  case class Params[L](weight: Tensor1[L, Float], bias: Tensor1[L, Float])

  object Params:
    def defaultInit[L: Label](ae: AxisExtent[L]) =
      Params(
        weight = Tensor(Shape(ae)).fill(1f),
        bias = Tensor(Shape(ae)).fill(0f)
      )
