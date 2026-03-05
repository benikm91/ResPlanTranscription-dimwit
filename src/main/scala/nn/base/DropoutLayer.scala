package resplan.nn.base

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Bernoulli

def sampleThinAffineLayer[In: Label, Out: Label](params: AffineLayer.Params[In, Out], dropoutRate: Float, key: Random.Key): AffineLayer.Params[In, Out] =
  val keepProb = 1.0f - dropoutRate
  val dropoutMask = IndependentDistribution.fromUnivariate(params.bias.shape, Bernoulli(Prob(keepProb))).sample(key).asFloat *! (1f / (keepProb))
  params.copy(
    weight = params.weight *! dropoutMask,
    bias = params.bias * dropoutMask
  )
