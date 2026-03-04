package resplan.nn

import dimwit.*
import nn.ActivationFunctions.gelu
import Util.vmap

trait ITransformerLayer[Context: Label, Embedding: Label] extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float]
  def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]

  def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    var x = context
    x = x + contextMixer(x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

trait ICrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label] extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):

  def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def transformerLayer: ITransformerLayer[Context, Embedding]

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    transformerLayer(context + crossContextMixer(crossContext, context))

case class TransformerBlock[Context: Label, Embedding](layers: List[ITransformerLayer[Context, Embedding]]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
  override def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(t):
      case (t, layer) => layer(t)

case class CrossTransformerBlock[CrossContext: Label, CrossEmbedding, Context: Label, Embedding](
    layers: List[CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]]
) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):
  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(context):
      case (t, layer) => layer(crossContext, context)

trait TransformerLayer[Context: Label, Embedding: Label](
    hyperParams: TransformerLayer.HyperParams[Context, Embedding]
)(
    params: TransformerLayer.Params[Embedding]
) extends ITransformerLayer[Context, Embedding]:

  val attentionPreNormalization = LayerNorm(params.attentionNormParams)
  val embeddingMixerPreNormalization = vmap(Axis[Context])(LayerNorm(params.embeddingMixerNormParams))

  val baseEmbeddingMixer = MLPEmbeddingMixer(hyperParams.embeddingMixer)(params.embeddingMixerParams)
  val baseContextMixer = MultiHeadAttention(hyperParams.multiHeadAttention)(params.multiHeadAttentionParams)

object TransformerLayer:

  case class WithPreNorm[Context: Label, Embedding: Label](
      hyperParams: TransformerLayer.HyperParams[Context, Embedding]
  )(
      params: TransformerLayer.Params[Embedding]
  ) extends TransformerLayer[Context, Embedding](hyperParams)(params):

    override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      (attentionPreNormalization andThen baseEmbeddingMixer)(embeddings)

    override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      (embeddingMixerPreNormalization andThen baseContextMixer)(context)

  case class WithPostNorm[Context: Label, Embedding: Label](
      hyperParams: TransformerLayer.HyperParams[Context, Embedding]
  )(
      params: TransformerLayer.Params[Embedding]
  ) extends TransformerLayer[Context, Embedding](hyperParams)(params):

    override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      (baseEmbeddingMixer andThen attentionPreNormalization)(embeddings)

    override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      (baseContextMixer andThen embeddingMixerPreNormalization)(context)

  case class HyperParams[Context: Label, Embedding: Label](
      embeddingMixer: MLPEmbeddingMixer.HyperParams[Embedding],
      multiHeadAttention: MultiHeadAttention.HyperParams[Context]
  )

  case class Params[Embedding](
      attentionNormParams: LayerNorm.Params[Embedding],
      multiHeadAttentionParams: MultiHeadAttention.Params[Embedding],
      embeddingMixerNormParams: LayerNorm.Params[Embedding],
      embeddingMixerParams: MLPEmbeddingMixer.Params[Embedding]
  )

  object Params:

    def defaultInit[E: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[E], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], numTransformerLayers: Int): Params[E] =
      val (attnKey, mixKey) = key.split2()
      new Params[E](
        attentionNormParams = LayerNorm.Params.defaultInit(embeddingExtent),
        multiHeadAttentionParams = MultiHeadAttention.Params.defaultInit(attnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, numTransformerLayers = numTransformerLayers),
        embeddingMixerNormParams = LayerNorm.Params.defaultInit(embeddingExtent),
        embeddingMixerParams = MLPEmbeddingMixer.Params.defaultInit(embeddingExtent, embeddingMixedExtent, mixKey)
      )

case class CrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
    hyperParams: CrossTransformerLayer.HyperParams[CrossContext, Context, Embedding]
)(
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding]
) extends ICrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]:

  private val multiHeadCrossAttention = new MultiHeadCrossAttention(hyperParams.multiHeadCrossAttention)(params.multiHeadCrossAttentionParams)

  override def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    multiHeadCrossAttention(crossContext, context)
  override val transformerLayer = new TransformerLayer.WithPostNorm(hyperParams.transformer)(params.transformerParams)

object CrossTransformerLayer:

  case class HyperParams[CrossContext: Label, Context: Label, Embedding: Label](
      transformer: TransformerLayer.HyperParams[Context, Embedding],
      multiHeadCrossAttention: MultiHeadCrossAttention.HyperParams[CrossContext, Context]
  )

  case class Params[CrossEmbedding, Embedding](
      multiHeadCrossAttentionParams: MultiHeadCrossAttention.Params[CrossEmbedding, Embedding],
      transformerParams: TransformerLayer.Params[Embedding]
  )

type EmbeddingMixer[E] = (Tensor1[E, Float] => Tensor1[E, Float])

case class MLPEmbeddingMixer[Embedding: Label](
    hyperParams: MLPEmbeddingMixer.HyperParams[Embedding]
)(
    params: MLPEmbeddingMixer.Params[Embedding]
) extends EmbeddingMixer[Embedding]:

  private val hiddenLayer = AffineLayer(params.c_fc)
  private val outputLayer = AffineLayer(params.c_proj)
  private val hiddenDropout = DropoutLayer(hyperParams.hiddenDropout)
  private val outputDropout = DropoutLayer(hyperParams.outputDropout)

  override def apply(in: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
    val hidden = hiddenDropout(gelu(hiddenLayer(in)))
    outputDropout(outputLayer(hidden))

object MLPEmbeddingMixer:

  trait EmbeddingMixed derives Label

  case class HyperParams[Embedding: Label](
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
