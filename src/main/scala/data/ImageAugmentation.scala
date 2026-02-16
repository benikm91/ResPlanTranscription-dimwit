package resplan.data

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Uniform
import dimwit.stats.Bernoulli
import dimwit.jax.Jax
import dimwit.stats.Categorical

object ImageAugmentation:

  import me.shadaj.scalapy.py
  private val jaxImage = py.module("jax.image")

  private trait A derives Label
  private val zoomFactorDist = IndependentDistribution.fromUnivariate((Shape1(Axis[A] -> 2)), Uniform(0.8f, 1.2f))
  private val relShiftDist = IndependentDistribution.fromUnivariate((Shape1(Axis[A] -> 2)), Uniform(-0.05f, 0.05f))
  private val flipHDist = Bernoulli(Prob(0.5f))
  private val flipVDist = Bernoulli(Prob(0.5f))
  private val rotationAngleDist = Uniform(Tensor0(0), Tensor0(3))

  def apply(image: Tensor3[Width, Height, Channel, Float], key: Random.Key): Tensor3[Width, Height, Channel, Float] =
    val keys = key.split(5)
    val width = image.shape(Axis[Width]).toFloat
    val height = image.shape(Axis[Height]).toFloat
    val zoomFactor = zoomFactorDist.sample(keys(0))
    val dim = Tensor1(Axis[A]).fromArray(Array(width, height))
    val shift = dim * relShiftDist.sample(keys(1))
    val center = dim /! 2f
    val totalTranslation = center - (center / zoomFactor) + shift
    val invImage = 1f -! image
    var x = jaxImage.scale_and_translate(
      invImage.jaxValue,
      shape = (160, 160, 3),
      spatial_dims = (0, 1),
      scale = zoomFactor.jaxValue,
      translation = totalTranslation.jaxValue,
      method = "bilinear"
    )

    val doFlipH = flipHDist.sample(keys(2))
    val doFlipV = flipVDist.sample(keys(3))

    x = Jax.jnp.where(doFlipH.jaxValue, Jax.jnp.fliplr(x), x)
    x = Jax.jnp.where(doFlipV.jaxValue, Jax.jnp.flipud(x), x)

    // Rotate 0, 90, 180, 270 degrees
    val rotation = rotationAngleDist.sample(keys(4))
    // TODO

    1f -! Tensor.fromPy(VType[Float])(x)
