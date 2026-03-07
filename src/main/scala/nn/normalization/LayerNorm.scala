package resplan.nn.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax

case class LayerNorm[L: Label](
    params: LayerNorm.Params[L],
    epsilon: Float
) extends (Tensor1[L, Float] => Tensor1[L, Float]):

  private def standardize(x: Tensor1[L, Float]): Tensor1[L, Float] =
    val x0 = x -! x.mean
    val variance = x0.pow(2).mean
    x0 /! (variance + epsilon).sqrt

  def apply(x: Tensor1[L, Float]): Tensor1[L, Float] =
    standardize(x) * params.weight + params.bias

object LayerNorm:

  def apply[L: Label](params: Params[L]): LayerNorm[L] =
    val epsilon = Jax.jnp.finfo(params.weight.dtype.jaxType).eps.as[Float]
    LayerNorm(params, epsilon)

  case class Params[L](weight: Tensor1[L, Float], bias: Tensor1[L, Float])

  object Params:
    def identity[L: Label](ae: AxisExtent[L]) =
      Params(
        weight = Tensor(Shape(ae)).fill(1f),
        bias = Tensor(Shape(ae)).fill(0f)
      )
