package resplan.data

import dimwit.*
import dimwit.Conversions.given
import nn.ActivationFunctions.*
import dimwit.jax.Jax
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import nn.Conv2DLayer
import dimwit.stats.Normal
import dimwit.stats.Bernoulli

object Util:
  def vmap[T <: Tuple: Labels, T2 <: Tuple: Labels, L: Label, V](axis: Axis[L])(f: (Tensor[T, V] => Tensor[T2, V])): Tensor[L *: T, V] => Tensor[L *: T2, V] =
    x => x.vmap(axis)(f)

  // TODO remove when dimwit core updated
  given listInstance[A](using ta: ToPyTree[A]): ToPyTree[List[A]] with
    def toPyTree(l: List[A]): Jax.PyAny =
      val pyItems = l.map(ta.toPyTree)
      py.Dynamic.global.list(pyItems.toPythonProxy)

    def fromPyTree(p: Jax.PyAny): List[A] =
      val pyList = p.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len): i =>
        ta.fromPyTree(pyList.bracketAccess(i))

import Util.vmap
import resplan.data as Context
import resplan.data as Embedding

case class DropoutLayer[L: Label](hyperParams: DropoutHyperParams[L]) extends (Tensor1[L, Float] => Tensor1[L, Float]):

  private def dropout(keepProb: Tensor0[Prob])(x: Tensor1[L, Float], key: Random.Key): Tensor1[L, Float] =
    val mask = IndependentDistribution.fromUnivariate(x.shape, Bernoulli(keepProb)).sample(key)
    x * mask.asFloat

  override def apply(x: Tensor1[L, Float]): Tensor1[L, Float] =
    import hyperParams.*
    if isTraining then
      // Scale output by 1/keepProb to maintain the expected value of the activations at inference time
      val keepProb = Prob(Tensor0(1f - dropoutRate))
      dropout(keepProb)(x, key) /! keepProb.asFloat
    else x

case class LayerNormalizationParams[L](
    weight: Tensor1[L, Float],
    bias: Tensor1[L, Float]
)

object LayerNormalizationParams:
  def init[L: Label](ae: AxisExtent[L]) =
    LayerNormalizationParams(
      weight = Tensor(Shape(ae)).fill(1f),
      bias = Tensor(Shape(ae)).fill(0f)
    )

case class LinearLayerParams[In, Out](
    weight: Tensor2[In, Out, Float],
    bias: Tensor1[Out, Float]
)

case class ProjectionLayerParams[In, Out](
    weight: Tensor2[In, Out, Float]
)
object ProjectionLayerParams:
  def init[In: Label, Out: Label](key: Random.Key, inExtent: AxisExtent[In], outExtent: AxisExtent[Out]): ProjectionLayerParams[In, Out] =
    ProjectionLayerParams(
      weight = Normal.standardIsotropic(Shape(inExtent, outExtent), scale = 0.02f).sample(key)
    )

case class HeadsParams[Kind, Embedding](val weights: Tensor3[Head, Embedding, Kind, Float])

case class MultiHeadAttentionParams[Embedding](
    wq: HeadsParams[HeadQuery, Embedding],
    wk: HeadsParams[HeadKey, Embedding],
    wv: HeadsParams[HeadValue, Embedding],
    proj: LinearLayerParams[Head |*| HeadValue, Embedding]
)

object MultiHeadAttentionParams:
  def init[Embedding: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding]): MultiHeadAttentionParams[Embedding] =
    MultiHeadAttentionParams(
      wq = HeadsParams(Normal.standardIsotropic(Shape(headExtent, embeddingExtent, headQueryExtent), scale = 0.02f).sample(key)),
      wk = HeadsParams(Normal.standardIsotropic(Shape(headExtent, embeddingExtent, headKeyExtent), scale = 0.02f).sample(key)),
      wv = HeadsParams(Normal.standardIsotropic(Shape(headExtent, embeddingExtent, headValueExtent), scale = 0.02f).sample(key)),
      proj = LinearLayerParams(
        weight = Normal.standardIsotropic(Shape(headExtent * headValueExtent, embeddingExtent), scale = 0.02f).sample(key),
        bias = Tensor(Shape(embeddingExtent)).fill(0f)
      )
    )

case class MultiHeadCrossAttentionParams[CrossEmbedding, Embedding](
    wq: HeadsParams[HeadQuery, Embedding],
    wk: HeadsParams[HeadKey, CrossEmbedding],
    wv: HeadsParams[HeadValue, CrossEmbedding],
    proj: LinearLayerParams[Head |*| HeadValue, Embedding]
)
object MultiHeadCrossAttentionParams:
  def init[CrossEmbedding: Label, Embedding: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding]): MultiHeadCrossAttentionParams[CrossEmbedding, Embedding] =
    MultiHeadCrossAttentionParams(
      wq = HeadsParams(Normal.standardIsotropic(Shape(headExtent, embeddingExtent, headQueryExtent), scale = 0.02f).sample(key)),
      wk = HeadsParams(Normal.standardIsotropic(Shape(headExtent, crossEmbeddingExtent, headKeyExtent), scale = 0.02f).sample(key)),
      wv = HeadsParams(Normal.standardIsotropic(Shape(headExtent, crossEmbeddingExtent, headValueExtent), scale = 0.02f).sample(key)),
      proj = LinearLayerParams(
        weight = Normal.standardIsotropic(Shape(headExtent * headValueExtent, embeddingExtent), scale = 0.02f).sample(key),
        bias = Tensor(Shape(embeddingExtent)).fill(0f)
      )
    )

case class EmbeddingMixerParams[Embedding](
    c_fc: LinearLayerParams[Embedding, EmbeddingMixed],
    c_proj: LinearLayerParams[EmbeddingMixed, Embedding]
)
object EmbeddingMixerParams:
  def init[Embedding: Label](key: Random.Key, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed]): EmbeddingMixerParams[Embedding] =
    val (fcKey, projKey) = key.split2()
    EmbeddingMixerParams(
      c_fc = LinearLayerParams(
        weight = Normal.standardIsotropic(Shape(embeddingExtent, embeddingMixedExtent), scale = 0.02f).sample(fcKey),
        bias = Tensor(Shape(embeddingMixedExtent)).fill(0f)
      ),
      c_proj = LinearLayerParams(
        weight = Normal.standardIsotropic(Shape(embeddingMixedExtent, embeddingExtent), scale = 0.02f).sample(projKey),
        bias = Tensor(Shape(embeddingExtent)).fill(0f)
      )
    )

case class TransformerLayerParams[Embedding](
    ln1: LayerNormalizationParams[Embedding],
    attn: MultiHeadAttentionParams[Embedding],
    ln2: LayerNormalizationParams[Embedding],
    embeddingMixer: EmbeddingMixerParams[Embedding]
)

object TransformerLayerParams:
  def init[E: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[E], embeddingMixedExtent: AxisExtent[EmbeddingMixed]): TransformerLayerParams[E] =
    val (attnKey, mixKey) = key.split2()
    TransformerLayerParams[E](
      ln1 = LayerNormalizationParams.init(embeddingExtent),
      attn = MultiHeadAttentionParams.init(attnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent),
      ln2 = LayerNormalizationParams.init(embeddingExtent),
      embeddingMixer = EmbeddingMixerParams.init(mixKey, embeddingExtent, embeddingMixedExtent)
    )

case class CrossTransformerLayerParams[CrossEmbedding, Embedding](
    crossAttentionPreNormalization: LayerNormalizationParams[Embedding],
    crossAttention: MultiHeadCrossAttentionParams[CrossEmbedding, Embedding],
    transformer: TransformerLayerParams[Embedding]
)

case class ViTPatchingParams(
    projectionWeights: Tensor[(Width, Height, Channel, EncoderEmbedding), Float]
)

object ViTPatchingParams:
  def init(key: Random.Key, patchWidthExtent: AxisExtent[Width], patchHeightExtent: AxisExtent[Height], channelExtent: AxisExtent[Channel], embeddingExtent: AxisExtent[EncoderEmbedding]): ViTPatchingParams =
    ViTPatchingParams(
      projectionWeights = Normal.standardIsotropic(Shape(patchWidthExtent, patchHeightExtent, channelExtent, embeddingExtent), scale = 0.02f).sample(key)
    )

case class LinearLayer[In: Label, Out: Label](params: LinearLayerParams[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
  override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
    x.dot(Axis[In])(params.weight) + params.bias

case class EmbeddingMixer[Embedding: Label](hyperParams: EmbeddingMixerHyperParams[Embedding])(params: EmbeddingMixerParams[Embedding]) extends (Tensor1[Embedding, Float] => Tensor1[Embedding, Float]):
  private val hiddenLayer = LinearLayer(params.c_fc)
  private val outputLayer = LinearLayer(params.c_proj)
  private val hiddenDropout = DropoutLayer(hyperParams.hiddenDropout)
  private val outputDropout = DropoutLayer(hyperParams.outputDropout)

  override def apply(in: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
    val hidden = hiddenDropout(gelu(hiddenLayer(in)))
    outputDropout(outputLayer(hidden))

case class ProjectionLayer[In: Label, Out: Label](params: ProjectionLayerParams[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
  def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] = x.dot(Axis[In])(params.weight)

def causalMasking[Context: Label](attnScores: Tensor2[Context, Prime[Context], Float]): Tensor2[Context, Prime[Context], Float] =
  val ctxLength = attnScores.shape(Axis[Context])
  val causalMask = tril(Tensor(Shape((Axis[Context] -> ctxLength, Axis[Prime[Context]] -> ctxLength))).fill(true))
  where(causalMask, attnScores, Tensor.like(attnScores).fill(Float.NegativeInfinity))

def noMasking[Context](attnScores: Tensor2[Context, Prime[Context], Float]): Tensor2[Context, Prime[Context], Float] = attnScores

object MultiHeadAttention:
  trait AttnWeights derives Label

case class MultiHeadAttention[Context: Label, Embedding: Label](
    hyperParams: MultiHeadAttentionHyperParams[Context]
)(
    params: MultiHeadAttentionParams[Embedding]
) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  import MultiHeadAttention.AttnWeights

  private val projection = LinearLayer(params.proj)
  private val attentionWeightDropout = DropoutLayer(hyperParams.attentionDropout)

  def apply(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    import hyperParams.*
    val heads = zipvmap(Axis[Head])(params.wq.weights, params.wk.weights, params.wv.weights):
      case (wq, wk, wv) =>
        attention(wq, wk, wv, attentionMasking)(x)
    heads.vmap(Axis[Context])(heads => projection(heads.flatten))

  private def attention(
      wq: Tensor2[Embedding, HeadQuery, Float],
      wk: Tensor2[Embedding, HeadKey, Float],
      wv: Tensor2[Embedding, HeadValue, Float],
      attentionMasking: Tensor[(Context, Prime[Context]), Float] => Tensor[(Context, Prime[Context]), Float]
  )(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, HeadValue, Float] =
    val queries = x.dot(Axis[Embedding])(wq)
    val keys = x.dot(Axis[Embedding])(wk)
    val values = x.dot(Axis[Embedding])(wv)
    val dk = Tensor0(Math.sqrt(keys.shape(Axis[HeadKey])).toFloat)
    val attnScores = (queries.dot(Axis[HeadQuery ~ HeadKey])(keys) /! dk)
    val attnWeights = attentionMasking(attnScores)
      .vmap(Axis[Context])(attnScore => softmax(attnScore).relabelTo(Axis[AttnWeights]))
    val droppedWeights = attnWeights.vmap(Axis[Context]): w =>
      attentionWeightDropout.apply(w)
    val res = droppedWeights.dot(Axis[AttnWeights ~ Context])(values)
    res

case class MultiHeadCrossAttentionHyperParams(
    crossAttentionWeightsDropout: DropoutHyperParams[MultiHeadCrossAttention.CrossAttnWeights]
)

object MultiHeadCrossAttention:
  trait CrossAttnWeights derives Label

case class MultiHeadCrossAttention[CrossEmbedding: Label, Embedding: Label](
    hyperParams: MultiHeadCrossAttentionHyperParams
)(
    params: MultiHeadCrossAttentionParams[CrossEmbedding, Embedding]
):

  import MultiHeadCrossAttention.CrossAttnWeights

  private val projection = LinearLayer(params.proj)
  private val crossAttentionWeightsDropout = DropoutLayer(hyperParams.crossAttentionWeightsDropout)

  def apply[CrossContext: Label, Context: Label](crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val heads = zipvmap(Axis[Head])(params.wq.weights, params.wk.weights, params.wv.weights):
      case (wq, wk, wv) =>
        attention(wq, wk, wv)(crossContext, context)
    heads.vmap(Axis[Context])(heads => projection(heads.flatten))

  private def attention[CrossContext: Label, Context: Label](
      wq: Tensor2[Embedding, HeadQuery, Float],
      wk: Tensor2[CrossEmbedding, HeadKey, Float],
      wv: Tensor2[CrossEmbedding, HeadValue, Float]
  )(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, HeadValue, Float] =
    val queries = context.dot(Axis[Embedding])(wq)
    val keys = crossContext.dot(Axis[CrossEmbedding])(wk)
    val values = crossContext.dot(Axis[CrossEmbedding])(wv)
    val dk = Tensor0(Math.sqrt(keys.shape(Axis[HeadKey])).toFloat)
    val crossAttnScores = (queries.dot(Axis[HeadQuery ~ HeadKey])(keys) /! dk)
    val crossAttnWeights = crossAttnScores
      .vmap(Axis[Context])(attnScore => softmax(attnScore).relabelTo(Axis[CrossAttnWeights]))
    val droppedCrossAttnWeights = crossAttnWeights.vmap(Axis[Context])(crossAttentionWeightsDropout)
    val res = droppedCrossAttnWeights.dot(Axis[CrossAttnWeights ~ CrossContext])(values)
    res

case class LayerNorm[L: Label](params: LayerNormalizationParams[L]) extends (Tensor1[L, Float] => Tensor1[L, Float]):

  private def standardize(x: Tensor1[L, Float]): Tensor1[L, Float] =
    val x0 = x -! x.mean
    val variance = x0.pow(2).mean
    val epsilon = 1e-6f
    x0 /! (variance + epsilon).sqrt

  def apply(x: Tensor1[L, Float]): Tensor1[L, Float] =
    standardize(x) * params.weight + params.bias

case class TransformerLayer[Context: Label, Embedding: Label](
    embeddingMixer: Tensor1[Embedding, Float] => Tensor1[Embedding, Float],
    contextMixer: Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]
) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    var x = t
    x = x + contextMixer(x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

case class DropoutHyperParams[L](
    dropoutRate: Float,
    key: Random.Key,
    isTraining: Boolean
):
  require(0f <= dropoutRate && dropoutRate <= 1f)

case class EmbeddingMixerHyperParams[Embedding](
    hiddenDropout: DropoutHyperParams[EmbeddingMixed],
    outputDropout: DropoutHyperParams[Embedding]
)

case class MultiHeadAttentionHyperParams[Context](
    attentionMasking: Tensor[(Context, Prime[Context]), Float] => Tensor[(Context, Prime[Context]), Float],
    attentionDropout: DropoutHyperParams[MultiHeadAttention.AttnWeights]
)

case class TransformerLayerHyperParams[Context, Embedding](
    embeddingMixer: EmbeddingMixerHyperParams[Embedding],
    attn: MultiHeadAttentionHyperParams[Context]
)

object TransformerLayer:
  def fromParams[Context: Label, Embedding: Label](
      hyperParams: TransformerLayerHyperParams[Context, Embedding]
  )(
      params: TransformerLayerParams[Embedding]
  ): TransformerLayer[Context, Embedding] =
    val preNormalization = LayerNorm(params.ln1)
    val postNormalization = vmap(Axis[Context])(LayerNorm(params.ln2))

    TransformerLayer(
      embeddingMixer = preNormalization andThen EmbeddingMixer(hyperParams.embeddingMixer)(params.embeddingMixer),
      contextMixer = postNormalization andThen MultiHeadAttention(hyperParams.attn)(params.attn)
    )

case class CrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
    crossContextMixer: (Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float],
    transformerLayer: TransformerLayer[Context, Embedding]
) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    transformerLayer(context + crossContextMixer(crossContext, context))

case class CrossTransformerLayerHyperParams[CrossContext, CrossEmbedding, Context, Embedding](
    crossAttention: MultiHeadCrossAttentionHyperParams,
    transformer: TransformerLayerHyperParams[Context, Embedding]
)

object CrossTransformerLayer:
  def fromParams[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
      hyperParams: CrossTransformerLayerHyperParams[CrossContext, CrossEmbedding, Context, Embedding]
  )(
      params: CrossTransformerLayerParams[CrossEmbedding, Embedding]
  ): CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding] =
    val crossAttentionPreNormalization = vmap(Axis[Context])(LayerNorm(params.crossAttentionPreNormalization))
    val multiHeadCrossAttention = MultiHeadCrossAttention(hyperParams.crossAttention)(params.crossAttention)
    CrossTransformerLayer(
      crossContextMixer = (crossContext, context) =>
        multiHeadCrossAttention(crossContext, crossAttentionPreNormalization(context)),
      TransformerLayer.fromParams(hyperParams.transformer)(params.transformer)
    )

case class TransformerBlock[Context: Label, Embedding](layers: List[TransformerLayer[Context, Embedding]]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
  override def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(t):
      case (t, layer) => layer(t)

case class CrossTransformerBlock[CrossContext: Label, CrossEmbedding, Context: Label, Embedding](
    layers: List[CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]]
) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):
  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(context):
      case (t, layer) => layer(crossContext, context)

case class ViTPatching(params: ViTPatchingParams):

  private val projection =
    val weightShape = params.projectionWeights.shape
    val kernelSize = (weightShape.extent(Axis[Width]), weightShape.extent(Axis[Height]))
    Conv2DLayer(Conv2DLayer.Params(params.projectionWeights), stride = kernelSize)

  private def positionalEncoding2D(shape: Shape3[Width, Height, EncoderEmbedding]): Tensor3[Width, Height, EncoderEmbedding, Float] =
    // 1. Prepare things we need for positional encoding
    val widthExtent = shape.extent(Axis[Width]).size
    val heightExtent = shape.extent(Axis[Height]).size
    val embedDim = shape.extent(Axis[EncoderEmbedding]).size
    val scaleCount = embedDim / 4
    val posScales = (Tensor1(Axis[EncoderEmbedding]).fromArray(Array.range(0, scaleCount)).asFloat *! -(Tensor0(10000.0f).log / scaleCount)).exp

    // 2. Prepare Width (X-axis)
    val widthPosRaw = Tensor1(Axis[Width]).fromArray(Array.range(0, widthExtent))
    val widthPosScaled = widthPosRaw.asFloat.vmap(Axis[Width])(_ *! posScales)
    val widthPosEncoded = concatenate(widthPosScaled.sin, widthPosScaled.cos, concatAxis = Axis[EncoderEmbedding])

    // 3. Prepare Height (Y-axis)
    val heightPosRaw = Tensor1(Axis[Height]).fromArray(Array.range(0, heightExtent))
    val heightPosScaled = heightPosRaw.asFloat.vmap(Axis[Height])(_ *! posScales)
    val heightPosEncoded = concatenate(heightPosScaled.sin, heightPosScaled.cos, concatAxis = Axis[EncoderEmbedding])

    // 4. Expansion into 2D Grids and Concatenation
    val widthPosGrid = stack(List.fill(heightExtent)(widthPosEncoded), newAxis = Axis[Height]).transpose(Axis[Width], Axis[Height], Axis[EncoderEmbedding])
    val heightPosGrid = stack(List.fill(widthExtent)(heightPosEncoded), newAxis = Axis[Width])

    concatenate(widthPosGrid, heightPosGrid, concatAxis = Axis[EncoderEmbedding])

  def apply(img: Tensor3[Width, Height, Channel, Float]): Tensor2[Width |*| Height, EncoderEmbedding, Float] =
    val projected = projection(img)
    val x = projection(img) + positionalEncoding2D(projected.shape)
    x.flatten((Axis[Width], Axis[Height]))

case class Timer private ():

  private var lastTime = System.currentTimeMillis()
  private var internalRunningAverage = -1f

  def tick(): Unit =
    val now = System.currentTimeMillis()
    val elapsed = now - lastTime
    internalRunningAverage =
      if internalRunningAverage == -1f
      then elapsed
      else 0.9f * internalRunningAverage + 0.1f * elapsed
    lastTime = now

  def reset(): Unit =
    lastTime = System.currentTimeMillis()
    internalRunningAverage = -1f

  def runningAvgSeconds: Float = internalRunningAverage / 1000f

case class Embedder[Context: Label, Embedding: Label](vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float], positionalEmbeddings: Tensor2[Context, Embedding, Float]):

  def apply(tokens: Tensor1[Context, Int]): Tensor2[Context, Embedding, Float] =
    val embeddings = vocabularyEmbeddings.take(Axis[Vocab])(tokens)
    embeddings + positionalEmbeddings

object Timer:
  def start(): Timer = new Timer()
