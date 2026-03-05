package resplan.nn.transformer

import dimwit.*
import nn.ActivationFunctions.gelu
import resplan.nn.normalization.LayerNorm

case class TransformerBlock[Context: Label, Embedding](layers: List[ITransformerLayer[Context, Embedding]]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
  override def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(t):
      case (t, layer) => layer(t)

trait ITransformerLayer[Context: Label, Embedding: Label] extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float]

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    var x = context
    x = x + contextMixer(x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

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
