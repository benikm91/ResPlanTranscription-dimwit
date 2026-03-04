package resplan.nn.base

import dimwit.*
import resplan.nn.init.xavierNormal

case class LinearLayer[In: Label, Out: Label](params: LinearLayer.Params[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
  override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
    x.dot(Axis[In])(params.weight)

object LinearLayer:

  def apply[In: Label, Out: Label](weight: Tensor2[In, Out, Float]): LinearLayer[In, Out] =
    LinearLayer(LinearLayer.Params(weight))

  case class Params[In, Out](weight: Tensor2[In, Out, Float])

  object Params:
    def defaultInit[In: Label, Out: Label](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key): Params[In, Out] =
      Params(weight = xavierNormal(Shape(inExtent, outExtent), key))
