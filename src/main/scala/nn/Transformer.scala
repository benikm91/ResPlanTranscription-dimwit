package resplan.nn

import dimwit.*
import nn.ActivationFunctions.gelu
import Util.vmap

trait ITransformerLayer[Context: Label, Embedding: Label] extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float]

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    var x = context
    x = x + contextMixer(x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

trait ICrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label] extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):

  def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float]

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    var x = context
    x = x + contextMixer(x)
    x = x + crossContextMixer(crossContext, x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

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

  val selfAttention = MultiHeadAttention(hyperParams.multiHeadAttention)(params.attentionParams)
  val selfAttentionNorm = LayerNorm(params.attentionNormParams)

  val mlp = MLPEmbeddingMixer(hyperParams.embeddingMixer)(params.mlpParams)
  val mlpNorm = LayerNorm(params.mlpNormParams)

object TransformerLayer:

  case class WithPreNorm[Context: Label, Embedding: Label](
      hyperParams: TransformerLayer.HyperParams[Context, Embedding]
  )(
      params: TransformerLayer.Params[Embedding]
  ) extends TransformerLayer[Context, Embedding](hyperParams)(params):

    override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      val embNorm = mlpNorm(embeddings)
      mlp(embNorm)

    override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      val contextNorm = context.vmap(Axis[Context])(selfAttentionNorm)
      selfAttention(contextNorm)

  case class WithPostNorm[Context: Label, Embedding: Label](
      hyperParams: TransformerLayer.HyperParams[Context, Embedding]
  )(
      params: TransformerLayer.Params[Embedding]
  ) extends TransformerLayer[Context, Embedding](hyperParams)(params):

    override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      mlpNorm(mlp(embeddings))

    override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      val mixed = selfAttention(context)
      mixed.vmap(Axis[Context])(selfAttentionNorm)

  case class HyperParams[Context: Label, Embedding: Label](
      embeddingMixer: MLPEmbeddingMixer.HyperParams[Embedding],
      multiHeadAttention: MultiHeadAttention.HyperParams[Context]
  )

  case class Params[Embedding](
      attentionParams: MultiHeadAttention.Params[Embedding],
      attentionNormParams: LayerNorm.Params[Embedding],
      mlpParams: MLPEmbeddingMixer.Params[Embedding],
      mlpNormParams: LayerNorm.Params[Embedding]
  )

  object Params:

    def defaultInit[E: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[E], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], numTransformerLayers: Int): Params[E] =
      val (attnKey, mixKey) = key.split2()
      new Params[E](
        attentionParams = MultiHeadAttention.Params.defaultInit(attnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, numTransformerLayers = numTransformerLayers),
        attentionNormParams = LayerNorm.Params.defaultInit(embeddingExtent),
        mlpParams = MLPEmbeddingMixer.Params.defaultInit(embeddingExtent, embeddingMixedExtent, mixKey),
        mlpNormParams = LayerNorm.Params.defaultInit(embeddingExtent)
      )

trait CrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
    hyperParams: CrossTransformerLayer.HyperParams[CrossContext, Context, Embedding]
)(
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding]
) extends ICrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]:

  val selfAttention = MultiHeadAttention(hyperParams.multiHeadAttention)(params.selfAttentionParams)
  val selfAttentionNorm = LayerNorm(params.selfAttentionNormParams)

  val crossAttention = new MultiHeadCrossAttention(hyperParams.multiHeadCrossAttention)(params.crossAttentionParams)
  val crossAttentionNorm = LayerNorm(params.crossAttentionNormParams)

  val mlp = MLPEmbeddingMixer(hyperParams.embeddingMixer)(params.mlpParams)
  val mlpNorm = LayerNorm(params.mlpNormParams)

object CrossTransformerLayer:

  case class WithPostNorm[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
      hyperParams: CrossTransformerLayer.HyperParams[CrossContext, Context, Embedding]
  )(
      params: CrossTransformerLayer.Params[CrossEmbedding, Embedding]
  ) extends CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding](hyperParams)(params):

    override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      val mixed = selfAttention(context)
      mixed.vmap(Axis[Context])(selfAttentionNorm)

    override def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      val mixed = crossAttention(crossContext, context)
      mixed.vmap(Axis[Context])(crossAttentionNorm)

    override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      val mixed = mlp(embeddings)
      mlpNorm(mixed)

  case class HyperParams[CrossContext: Label, Context: Label, Embedding: Label](
      embeddingMixer: MLPEmbeddingMixer.HyperParams[Embedding],
      multiHeadAttention: MultiHeadAttention.HyperParams[Context],
      multiHeadCrossAttention: MultiHeadCrossAttention.HyperParams[CrossContext, Context]
  )

  case class Params[CrossEmbedding, Embedding](
      crossAttentionParams: MultiHeadCrossAttention.Params[CrossEmbedding, Embedding],
      crossAttentionNormParams: LayerNorm.Params[Embedding],
      selfAttentionParams: MultiHeadAttention.Params[Embedding],
      selfAttentionNormParams: LayerNorm.Params[Embedding],
      mlpNormParams: LayerNorm.Params[Embedding],
      mlpParams: MLPEmbeddingMixer.Params[Embedding]
  )

  object Params:

    def defaultInit[CE: Label, E: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CE], embeddingExtent: AxisExtent[E], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], numTransformerLayers: Int): Params[CE, E] =
      val (attnKey, mixKey) = key.split2()
      new Params[CE, E](
        crossAttentionParams = MultiHeadCrossAttention.Params.defaultInit(attnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, crossEmbeddingExtent, embeddingExtent),
        crossAttentionNormParams = LayerNorm.Params.defaultInit(embeddingExtent),
        selfAttentionParams = MultiHeadAttention.Params.defaultInit(attnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, numTransformerLayers = numTransformerLayers),
        selfAttentionNormParams = LayerNorm.Params.defaultInit(embeddingExtent),
        mlpParams = MLPEmbeddingMixer.Params.defaultInit(embeddingExtent, embeddingMixedExtent, mixKey),
        mlpNormParams = LayerNorm.Params.defaultInit(embeddingExtent)
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
