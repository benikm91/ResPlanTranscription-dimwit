package resplan.nn.base

import dimwit.*
import resplan.nn.init.xavierNormal

case class AffineLayer[In: Label, Out: Label](params: AffineLayer.Params[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
  override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
    x.dot(Axis[In])(params.weight) + params.bias

object AffineLayer:
  case class Params[In, Out](
      weight: Tensor2[In, Out, Float],
      bias: Tensor1[Out, Float]
  )

  object Params:
    def defaultInit[In: Label, Out: Label](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key): Params[In, Out] =
      Params(
        weight = xavierNormal(Shape(inExtent, outExtent), key),
        bias = Tensor(Shape(outExtent)).fill(0f)
      )
