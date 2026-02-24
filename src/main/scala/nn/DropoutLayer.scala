package resplan.nn

import dimwit.*
import dimwit.stats.Bernoulli

case class DropoutLayer[L: Label](hyperParams: DropoutLayer.HyperParams[L]) extends (Tensor1[L, Float] => Tensor1[L, Float]):

  private def dropout(keepProb: Tensor0[Prob])(x: Tensor1[L, Float], key: Random.Key): Tensor1[L, Float] =
    val mask = IndependentDistribution.fromUnivariate(x.shape, Bernoulli(keepProb)).sample(key)
    x * mask.asFloat

  override def apply(x: Tensor1[L, Float]): Tensor1[L, Float] =
    import hyperParams.*
    if applyDropout then
      // Scale output by 1/keepProb to maintain the expected value of the activations at inference time
      val keepProb = Prob(Tensor0(1f - dropoutRate))
      dropout(keepProb)(x, sourceOfRandomness.next()) /! keepProb.asFloat
    else x

object DropoutLayer:
  case class HyperParams[L](
      dropoutRate: Float,
      sourceOfRandomness: Iterator[Random.Key], // TODO this is not pure
      applyDropout: Boolean
  ):
    require(0f <= dropoutRate && dropoutRate <= 1f)
