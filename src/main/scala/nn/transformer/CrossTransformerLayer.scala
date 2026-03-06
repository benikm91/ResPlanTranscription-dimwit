package resplan.nn.transformer

import dimwit.*
import nn.ActivationFunctions.gelu
import resplan.nn.normalization.LayerNorm

case class CrossTransformerBlock[CrossContext: Label, CrossEmbedding, Context: Label, Embedding](
    layers: List[CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]]
) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):
  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)

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

class CrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
    hyperParams: CrossTransformerLayer.HyperParams[CrossContext, Context, Embedding]
)(
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding]
) extends ICrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]:

  private val selfAttention = MultiHeadAttention(hyperParams.multiHeadAttention)(params.selfAttentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNormParams)

  private val crossAttention = new MultiHeadCrossAttention(hyperParams.multiHeadCrossAttention)(params.crossAttentionParams)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNormParams)

  private val mlp = MLPEmbeddingMixer(hyperParams.embeddingMixer)(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
    mlp(mlpPreNorm(embeddings))

  override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

  override def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    crossAttention(crossContext, context.vmap(Axis[Context])(crossAttentionPreNorm))

object CrossTransformerLayer:

  def apply[CrossContext: Label, Context: Label, CrossEmbedding: Label, Embedding: Label](hyperParams: HyperParams[CrossContext, Context, Embedding])(params: Params[CrossEmbedding, Embedding]): CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding] =
    new CrossTransformerLayer(hyperParams)(params)

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
      val (selfAttnKey, crossAttnKey, mixKey) = key.splitToTuple(3)
      new Params[CE, E](
        crossAttentionParams = MultiHeadCrossAttention.Params.defaultInit(crossAttnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, crossEmbeddingExtent, embeddingExtent, numTransformerLayers),
        crossAttentionNormParams = LayerNorm.Params.defaultInit(embeddingExtent),
        selfAttentionParams = MultiHeadAttention.Params.defaultInit(selfAttnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, numTransformerLayers),
        selfAttentionNormParams = LayerNorm.Params.defaultInit(embeddingExtent),
        mlpParams = MLPEmbeddingMixer.Params.defaultInit(embeddingExtent, embeddingMixedExtent, mixKey),
        mlpNormParams = LayerNorm.Params.defaultInit(embeddingExtent)
      )
