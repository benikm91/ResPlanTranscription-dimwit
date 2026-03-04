package resplan.nn.init

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal

def xavierScale(fanIn: Int, fanOut: Int): Float =
  Math.sqrt(2.0 / (fanIn + fanOut).toDouble).toFloat

def xavierNormal[FanIn: Label, FanOut: Label](shape: Shape2[FanIn, FanOut], key: Random.Key): Tensor2[FanIn, FanOut, Float] =
  val scale = xavierScale(shape(Axis[FanIn]), shape(Axis[FanOut]))
  Normal.standardIsotropic(shape, scale = scale).sample(key)
