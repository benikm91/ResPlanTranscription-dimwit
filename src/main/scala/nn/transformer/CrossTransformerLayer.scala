package resplan.nn.transformer

import dimwit.*
import nn.ActivationFunctions.gelu
import resplan.nn.normalization.LayerNorm

case class CrossTransformerBlock[CrossContext: Label, CrossEmbedding, Context: Label, Embedding](
    layers: List[CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]]
) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):
  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(context):
      case (t, layer) => layer(crossContext, context)

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
