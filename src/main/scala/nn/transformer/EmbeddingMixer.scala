package resplan.nn.transformer

import dimwit.*
import nn.ActivationFunctions.gelu
import resplan.nn.base.{DropoutLayer, AffineLayer}

case class MLPEmbeddingMixer[Embedding: Label](
    hyperParams: MLPEmbeddingMixer.HyperParams[Embedding]
)(
    params: MLPEmbeddingMixer.Params[Embedding]
) extends (Tensor1[Embedding, Float] => Tensor1[Embedding, Float]):

  private val hiddenLayer = AffineLayer(params.c_fc)
  private val outputLayer = AffineLayer(params.c_proj)
  private val hiddenDropout = DropoutLayer(hyperParams.hiddenDropout)
  private val outputDropout = DropoutLayer(hyperParams.outputDropout)

  override def apply(in: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
    val hidden = hiddenDropout(gelu(hiddenLayer(in)))
    outputDropout(outputLayer(hidden))

object MLPEmbeddingMixer:

  trait EmbeddingMixed derives Label

  case class HyperParams[Embedding](
      hiddenDropout: DropoutLayer.HyperParams[EmbeddingMixed],
      outputDropout: DropoutLayer.HyperParams[Embedding],
      activationFunction: Tensor[Tuple1[Embedding], Float] => Tensor[Tuple1[Embedding], Float]
  )

  object HyperParams:
    def apply[Embedding: Label](
        hiddenDropout: DropoutLayer.HyperParams[EmbeddingMixed],
        outputDropout: DropoutLayer.HyperParams[Embedding]
    ) = new HyperParams(hiddenDropout, outputDropout, gelu)

  case class Params[Embedding](
      c_fc: AffineLayer.Params[Embedding, EmbeddingMixed],
      c_proj: AffineLayer.Params[EmbeddingMixed, Embedding]
  )

  object Params:
    def defaultInit[Embedding: Label](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Random.Key): Params[Embedding] =
      val (fcKey, projKey) = key.split2()
      Params(
        c_fc = AffineLayer.Params.defaultInit(embeddingExtent, embeddingMixedExtent, fcKey),
        c_proj = AffineLayer.Params.defaultInit(embeddingMixedExtent, embeddingExtent, projKey)
      )
