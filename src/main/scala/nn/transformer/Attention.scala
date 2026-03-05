package resplan.nn.transformer

import dimwit.*
import dimwit.Conversions.given
import nn.ActivationFunctions.softmax
import dimwit.stats.Normal
import resplan.nn.base.{AffineLayer, LinearLayer}
import resplan.nn.init.xavierNormal

trait Query derives Label
trait Key derives Label
trait Value derives Label
trait AttentionWeights derives Label

trait ISelfAttention[Context: Label, Embedding: Label, Q: Label, K: Label, V: Label] extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, V, Float]):

  def encodeToQuery(embedding: Tensor1[Embedding, Float]): Tensor1[Q, Float]
  def encodeToKey(embedding: Tensor1[Embedding, Float]): Tensor1[K, Float]
  def encodeToValue(embedding: Tensor1[Embedding, Float]): Tensor1[V, Float]
  def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[Context, K, Float]): Tensor2[Context, AttentionWeights, Float]

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, V, Float] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = context.vmap(Axis[Context])(encodeToKey)
    val values = context.vmap(Axis[Context])(encodeToValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    val res = attentionWeights.dot(Axis[AttentionWeights ~ Context])(values)
    res

case class SelfAttention[Context: Label, Embedding: Label, Q: Label, K: Label, V: Label](
    hyperParams: SelfAttention.HyperParams[Context]
)(
    params: SelfAttention.BaseParams[Embedding, Q, K, V]
) extends ISelfAttention[Context, Embedding, Q, K, V]:

  override def encodeToQuery(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wq)(embedding)
  override def encodeToKey(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wk)(embedding)
  override def encodeToValue(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wv)(embedding)

  def calculateAttentionScores(queries: Tensor2[Context, Q, Float], keys: Tensor2[Context, K, Float]): Tensor2[Context, Prime[Context], Float] =
    val dk = Math.sqrt(keys.shape(Axis[K])).toFloat
    queries.dot(Axis[Q ~ K])(keys) /! dk

  override def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[Context, K, Float]) =
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = where(hyperParams.createAttentionMask(attentionScores.shape), attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
      .vmap(Axis[Context])(attentionScore => softmax(attentionScore).relabelTo(Axis[AttentionWeights]))
    attentionWeights

object SelfAttention:

  case class HyperParams[Context](
      createAttentionMask: Shape2[Context, Prime[Context]] => Tensor[(Context, Prime[Context]), Boolean]
      // calculateAttentionWeights: (Tensor2[Context, Q, Float], Tensor2[Context, K, Float]) => Tensor2[Context, AttentionWeights, Float]
  )

  case class BaseParams[Embedding, Q, K, V](
      wq: LinearLayer.Params[Embedding, Q],
      wk: LinearLayer.Params[Embedding, K],
      wv: LinearLayer.Params[Embedding, V]
  )

  object BaseParams:
    def apply[E, Q, K, V](wq: Tensor2[E, Q, Float], wk: Tensor2[E, K, Float], wv: Tensor2[E, V, Float]): BaseParams[E, Q, K, V] =
      BaseParams(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[Embedding: Label, Q: Label, K: Label, V: Label](queryExtent: AxisExtent[Q], keyExtent: AxisExtent[K], valueExtent: AxisExtent[V], embeddingExtent: AxisExtent[Embedding], key: Random.Key): BaseParams[Embedding, Q, K, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      BaseParams(
        wq = LinearLayer.Params.defaultInit(embeddingExtent, queryExtent, queryKey),
        wk = LinearLayer.Params.defaultInit(embeddingExtent, keyExtent, keyKey),
        wv = LinearLayer.Params.defaultInit(embeddingExtent, valueExtent, valueKey)
      )

trait ICrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label, Q: Label, K: Label, V: Label] extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, V, Float]):

  def encodeToQuery(embedding: Tensor1[Embedding, Float]): Tensor1[Q, Float]
  def encodeToKey(embedding: Tensor1[CrossEmbedding, Float]): Tensor1[K, Float]
  def encodeToValue(embedding: Tensor1[CrossEmbedding, Float]): Tensor1[V, Float]
  def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[CrossContext, K, Float]): Tensor2[Context, AttentionWeights, Float]

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, V, Float] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = crossContext.vmap(Axis[CrossContext])(encodeToKey)
    val values = crossContext.vmap(Axis[CrossContext])(encodeToValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    val res = attentionWeights.dot(Axis[AttentionWeights ~ CrossContext])(values)
    res

case class CrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label, Q: Label, K: Label, V: Label](
    hyperParams: CrossAttention.HyperParams[CrossContext, Context]
)(
    params: CrossAttention.BaseParams[CrossEmbedding, Embedding, Q, K, V]
) extends ICrossAttention[CrossContext, CrossEmbedding, Context, Embedding, Q, K, V]:

  override def encodeToQuery(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wq)(embedding)
  override def encodeToKey(embedding: Tensor1[CrossEmbedding, Float]) = LinearLayer(params.wk)(embedding)
  override def encodeToValue(embedding: Tensor1[CrossEmbedding, Float]) = LinearLayer(params.wv)(embedding)

  def calculateAttentionScores(queries: Tensor2[Context, Q, Float], keys: Tensor2[CrossContext, K, Float]): Tensor2[Context, CrossContext, Float] =
    val dk = Math.sqrt(keys.shape(Axis[K])).toFloat
    queries.dot(Axis[Q ~ K])(keys) /! dk

  override def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[CrossContext, K, Float]) =
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = where(hyperParams.createAttentionMask(attentionScores.shape), attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
      .vmap(Axis[Context])(attentionScore => softmax(attentionScore).relabelTo(Axis[AttentionWeights]))
    attentionWeights

object CrossAttention:

  case class HyperParams[CrossContext, Context](
      createAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Boolean]
  )

  case class BaseParams[CrossEmbedding, Embedding, Q, K, V](
      wq: LinearLayer.Params[Embedding, Q],
      wk: LinearLayer.Params[CrossEmbedding, K],
      wv: LinearLayer.Params[CrossEmbedding, V]
  )

  object BaseParams:

    def apply[CE, E, Q, K, V](wq: Tensor2[E, Q, Float], wk: Tensor2[CE, K, Float], wv: Tensor2[CE, V, Float]): BaseParams[CE, E, Q, K, V] =
      new BaseParams(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[CrossEmbedding: Label, Embedding: Label, Q: Label, K: Label, V: Label](queryExtent: AxisExtent[Q], keyExtent: AxisExtent[K], valueExtent: AxisExtent[V], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], key: Random.Key): BaseParams[CrossEmbedding, Embedding, Q, K, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      BaseParams(
        wq = LinearLayer.Params.defaultInit(embeddingExtent, queryExtent, queryKey),
        wk = LinearLayer.Params.defaultInit(crossEmbeddingExtent, keyExtent, keyKey),
        wv = LinearLayer.Params.defaultInit(crossEmbeddingExtent, valueExtent, valueKey)
      )

trait Head derives Label
trait HeadQuery derives Label
trait HeadKey derives Label
trait HeadValue derives Label

trait IMultiHeadSelfAttention[Context: Label, Embedding: Label] extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  def headAttention(context: Tensor2[Context, Embedding, Float]): Tensor[(Head, Context, HeadValue), Float]
  def headProjection(headValues: Tensor1[Head |*| HeadValue, Float]): Tensor1[Embedding, Float]

  override def apply(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val heads = headAttention(x)
    heads.vmap(Axis[Context])(heads => headProjection(heads.flatten))

case class MultiHeadAttention[Context: Label, Embedding: Label](
    hyperParams: MultiHeadAttention.HyperParams[Context]
)(
    params: MultiHeadAttention.Params[Embedding]
) extends IMultiHeadSelfAttention[Context, Embedding]:

  private val headProjectionLayer = AffineLayer(params.headProjection)

  override def headAttention(context: Tensor2[Context, Embedding, Float]): Tensor[(Head, Context, HeadValue), Float] =
    zipvmap(Axis[Head])(params.wq, params.wk, params.wv):
      case (wq, wk, wv) =>
        val attention = SelfAttention(hyperParams.headAttention)(SelfAttention.BaseParams(wq, wk, wv))
        attention(context)

  override def headProjection(headValues: Tensor1[Head |*| HeadValue, Float]) = headProjectionLayer(headValues)

object MultiHeadAttention:

  case class HyperParams[Context](
      headAttention: SelfAttention.HyperParams[Context]
  )

  case class Params[Embedding](
      wq: Tensor3[Head, Embedding, HeadQuery, Float],
      wk: Tensor3[Head, Embedding, HeadKey, Float],
      wv: Tensor3[Head, Embedding, HeadValue, Float],
      headProjection: AffineLayer.Params[Head |*| HeadValue, Embedding]
  )

  object Params:
    def defaultInit[Embedding: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding], numTransformerLayers: Int): Params[Embedding] =
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val nHeads = headExtent.size
      Params(
        wq = stack(queryKey.split(nHeads).map(key => xavierNormal(Shape(embeddingExtent, headQueryExtent), key)), Axis[Head]),
        wk = stack(keyKey.split(nHeads).map(key => xavierNormal(Shape(embeddingExtent, headKeyExtent), key)), Axis[Head]),
        wv = stack(valueKey.split(nHeads).map(key => xavierNormal(Shape(embeddingExtent, headValueExtent), key)), Axis[Head]),
        headProjection = AffineLayer.Params(
          weight = xavierNormal(Shape(headExtent * headValueExtent, embeddingExtent), projectionKey) /! (2f * numTransformerLayers).sqrt,
          bias = Tensor(Shape(embeddingExtent)).fill(0f)
        )
      )

trait IMultiHeadCrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label] extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):

  def headAttention(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor[(Head, Context, HeadValue), Float]
  def headProjection(headValues: Tensor1[Head |*| HeadValue, Float]): Tensor1[Embedding, Float]

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val heads = headAttention(crossContext, context)
    heads.vmap(Axis[Context])(heads => headProjection(heads.flatten))

case class MultiHeadCrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
    hyperParams: MultiHeadCrossAttention.HyperParams[CrossContext, Context]
)(
    params: MultiHeadCrossAttention.Params[CrossEmbedding, Embedding]
) extends IMultiHeadCrossAttention[CrossContext, CrossEmbedding, Context, Embedding]:

  override def headAttention(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]) =
    zipvmap(Axis[Head])(params.wq, params.wk, params.wv):
      case (wq, wk, wv) =>
        val headAttention = CrossAttention(hyperParams.headAttention)(CrossAttention.BaseParams(wq, wk, wv))
        headAttention(crossContext, context)

  override def headProjection(headValues: Tensor1[Head |*| HeadValue, Float]) = AffineLayer(params.headProjection)(headValues)

object MultiHeadCrossAttention:

  case class HyperParams[CrossContext, Context](
      val headAttention: CrossAttention.HyperParams[CrossContext, Context]
  )

  case class Params[CrossEmbedding, Embedding](
      wq: Tensor3[Head, Embedding, HeadQuery, Float],
      wk: Tensor3[Head, CrossEmbedding, HeadKey, Float],
      wv: Tensor3[Head, CrossEmbedding, HeadValue, Float],
      headProjection: AffineLayer.Params[Head |*| HeadValue, Embedding]
  )

  object Params:

    def defaultInit[CrossEmbedding: Label, Embedding: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding]): MultiHeadCrossAttention.Params[CrossEmbedding, Embedding] =
      MultiHeadCrossAttention.Params(
        wq = Normal.standardIsotropic(Shape(headExtent, embeddingExtent, headQueryExtent), scale = 0.02f).sample(key),
        wk = Normal.standardIsotropic(Shape(headExtent, crossEmbeddingExtent, headKeyExtent), scale = 0.02f).sample(key),
        wv = Normal.standardIsotropic(Shape(headExtent, crossEmbeddingExtent, headValueExtent), scale = 0.02f).sample(key),
        headProjection = AffineLayer.Params(
          weight = Normal.standardIsotropic(Shape(headExtent * headValueExtent, embeddingExtent), scale = 0.02f).sample(key),
          bias = Tensor(Shape(embeddingExtent)).fill(0f)
        )
      )
