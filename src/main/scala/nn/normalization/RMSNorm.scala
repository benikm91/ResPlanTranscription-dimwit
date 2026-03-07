package resplan.nn.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax

case class RMSNorm[L: Label](
    params: RMSNorm.Params[L],
    epsilon: Float
) extends (Tensor1[L, Float] => Tensor1[L, Float]):

  private def rescale(x: Tensor1[L, Float]): Tensor1[L, Float] =
    val variance = (x -! x.mean).pow(2).mean
    x /! (variance + epsilon).sqrt

  def apply(x: Tensor1[L, Float]): Tensor1[L, Float] =
    rescale(x) * params.weight

object RMSNorm:

  def apply[L: Label](params: Params[L]): RMSNorm[L] =
    val epsilon = Jax.jnp.finfo(params.weight.dtype.jaxType).eps.as[Float]
    RMSNorm(params, epsilon)

  case class Params[L](weight: Tensor1[L, Float])

  object Params:

    def identity[L: Label](ae: AxisExtent[L]) =
      Params(weight = Tensor(Shape(ae)).fill(1f))
