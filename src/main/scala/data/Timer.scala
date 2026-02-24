package resplan.data

/**  *
  * import dimwit.*
  * import dimwit.Conversions.given
  * import nn.ActivationFunctions.*
  * import dimwit.jax.Jax
  * import me.shadaj.scalapy.py
  * import me.shadaj.scalapy.py.SeqConverters
  * import nn.Conv2DLayer
  * import dimwit.stats.Normal
  * import dimwit.stats.Bernoulli
  * import resplan.nn.AffineLayer
  * import resplan.nn.xavierNormal
  * import resplan.nn.LayerNorm
  * import resplan.nn.DropoutLayer
  *
  * object Util:
  *  def vmap[T <: Tuple: Labels, T2 <: Tuple: Labels, L: Label, V](axis: Axis[L])(f: (Tensor[T, V] => Tensor[T2, V])): Tensor[L *: T, V] => Tensor[L *: T2, V] =
  *    x => x.vmap(axis)(f)
  *
  * import Util.vmap
  *
  * case class HeadsParams[Kind, Embedding](val weights: Tensor3[Head, Embedding, Kind, Float])
  *
  * case class MultiHeadAttentionParams[Embedding](
  *    wq: HeadsParams[HeadQuery, Embedding],
  *    wk: HeadsParams[HeadKey, Embedding],
  *    wv: HeadsParams[HeadValue, Embedding],
  *    proj: AffineLayer.Params[Head |*| HeadValue, Embedding]
  * )
  *
  * object MultiHeadAttentionParams:
  *  def init[Embedding: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding], numTransformerLayers: Int): MultiHeadAttentionParams[Embedding] =
  *    val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
  *    val nHeads = headExtent.size
  *
  *    MultiHeadAttentionParams(
  *      wq = HeadsParams(stack(queryKey.split(nHeads).map(key => xavierNormal(Shape(embeddingExtent, headQueryExtent), key)), Axis[Head])),
  *      wk = HeadsParams(stack(keyKey.split(nHeads).map(key => xavierNormal(Shape(embeddingExtent, headKeyExtent), key)), Axis[Head])),
  *      wv = HeadsParams(stack(valueKey.split(nHeads).map(key => xavierNormal(Shape(embeddingExtent, headValueExtent), key)), Axis[Head])),
  *      proj = AffineLayer.Params(
  *        weight = xavierNormal(Shape(headExtent * headValueExtent, embeddingExtent), projectionKey) /! (2f * numTransformerLayers).sqrt,
  *        bias = Tensor(Shape(embeddingExtent)).fill(0f)
  *      )
  *    )
  *
  * case class MultiHeadCrossAttentionParams[CrossEmbedding, Embedding](
  *    wq: HeadsParams[HeadQuery, Embedding],
  *    wk: HeadsParams[HeadKey, CrossEmbedding],
  *    wv: HeadsParams[HeadValue, CrossEmbedding],
  *    proj: AffineLayer.Params[Head |*| HeadValue, Embedding]
  * )
  * object MultiHeadCrossAttentionParams:
  *  def init[CrossEmbedding: Label, Embedding: Label](key: Random.Key, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding]): MultiHeadCrossAttentionParams[CrossEmbedding, Embedding] =
  *    MultiHeadCrossAttentionParams(
  *      wq = HeadsParams(Normal.standardIsotropic(Shape(headExtent, embeddingExtent, headQueryExtent), scale = 0.02f).sample(key)),
  *      wk = HeadsParams(Normal.standardIsotropic(Shape(headExtent, crossEmbeddingExtent, headKeyExtent), scale = 0.02f).sample(key)),
  *      wv = HeadsParams(Normal.standardIsotropic(Shape(headExtent, crossEmbeddingExtent, headValueExtent), scale = 0.02f).sample(key)),
  *      proj = AffineLayer.Params(
  *        weight = Normal.standardIsotropic(Shape(headExtent * headValueExtent, embeddingExtent), scale = 0.02f).sample(key),
  *        bias = Tensor(Shape(embeddingExtent)).fill(0f)
  *      )
  *    )
  *
  * case class EmbeddingMixerParams[Embedding](
  *    c_fc: AffineLayer.Params[Embedding, EmbeddingMixed],
  *    c_proj: AffineLayer.Params[EmbeddingMixed, Embedding]
  * )
  * object EmbeddingMixerParams:
  *  def defaultInit[Embedding: Label](key: Random.Key, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed]): EmbeddingMixerParams[Embedding] =
  *    val (fcKey, projKey) = key.split2()
  *    EmbeddingMixerParams(
  *      c_fc = AffineLayer.Params.defaultInit(embeddingExtent, embeddingMixedExtent, fcKey),
  *      c_proj = AffineLayer.Params.defaultInit(embeddingMixedExtent, embeddingExtent, projKey)
  *    )
  *
  * case class CrossTransformerLayerParams[CrossEmbedding, Embedding](
  *    crossAttentionPreNormalization: LayerNorm.Params[Embedding],
  *    crossAttention: BasicMultiHeadCrossAttention.Params[CrossEmbedding, Embedding],
  *    transformer: TransformerLayerParams[Embedding]
  * )
  *
  * case class ViTPatchingParams(
  *    projectionWeights: Tensor[(Width, Height, Channel, EncoderEmbedding), Float]
  * )
  *
  * object ViTPatchingParams:
  *  def init(key: Random.Key, patchWidthExtent: AxisExtent[Width], patchHeightExtent: AxisExtent[Height], channelExtent: AxisExtent[Channel], embeddingExtent: AxisExtent[EncoderEmbedding]): ViTPatchingParams =
  *    ViTPatchingParams(
  *      projectionWeights = Normal.standardIsotropic(Shape(patchWidthExtent, patchHeightExtent, channelExtent, embeddingExtent), scale = 0.02f).sample(key)
  *    )
  *
  * case class EmbeddingMixer[Embedding: Label](hyperParams: EmbeddingMixerHyperParams[Embedding])(params: EmbeddingMixerParams[Embedding]) extends (Tensor1[Embedding, Float] => Tensor1[Embedding, Float]):
  *  private val hiddenLayer = AffineLayer(params.c_fc)
  *  private val outputLayer = AffineLayer(params.c_proj)
  *  private val hiddenDropout = DropoutLayer(hyperParams.hiddenDropout)
  *  private val outputDropout = DropoutLayer(hyperParams.outputDropout)
  *
  *  override def apply(in: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
  *    val hidden = hiddenDropout(gelu(hiddenLayer(in)))
  *    outputDropout(outputLayer(hidden))
  *
  * def causalMask[Context: Label, CrossContext: Label](scoreShape: Shape2[Context, CrossContext]): Tensor[(Context, CrossContext), Boolean] =
  *  tril(noMask(scoreShape))
  *
  * def noMask[Context: Label, CrossContext: Label](scoreShape: Shape2[Context, CrossContext]): Tensor[(Context, CrossContext), Boolean] =
  *  Tensor(scoreShape).fill(true)
  *
  * object MultiHeadAttentionOld:
  *  trait AttnWeights derives Label
  *
  * case class MultiHeadAttentionOld[Context: Label, Embedding: Label](
  *    hyperParams: MultiHeadAttentionHyperParams[Context]
  * )(
  *    params: MultiHeadAttentionParams[Embedding]
  * ) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
  *
  *  import MultiHeadAttentionOld.AttnWeights
  *
  *  private val projection = AffineLayer(params.proj)
  *  private val attentionWeightDropout = DropoutLayer(hyperParams.attentionDropout)
  *
  *  def apply(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
  *    import hyperParams.*
  *    val heads = zipvmap(Axis[Head])(params.wq.weights, params.wk.weights, params.wv.weights):
  *      case (wq, wk, wv) =>
  *        attention(wq, wk, wv, createAttentionMasking)(x)
  *    heads.vmap(Axis[Context])(heads => ???)
  *
  *  private def attention(
  *      wq: Tensor2[Embedding, HeadQuery, Float],
  *      wk: Tensor2[Embedding, HeadKey, Float],
  *      wv: Tensor2[Embedding, HeadValue, Float],
  *      createAttentionMasking: AxisExtent[Context] => Tensor[(Context, Prime[Context]), Boolean]
  *  )(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, HeadValue, Float] =
  *    val queries = x.dot(Axis[Embedding])(wq)
  *    val keys = x.dot(Axis[Embedding])(wk)
  *    val values = x.dot(Axis[Embedding])(wv)
  *    val dk = Tensor0(Math.sqrt(keys.shape(Axis[HeadKey])).toFloat)
  *    val attnScores = (queries.dot(Axis[HeadQuery ~ HeadKey])(keys) /! dk)
  *    val attnWeights = where(createAttentionMasking(x.shape.extent(Axis[Context])), attnScores, Tensor.like(attnScores).fill(Float.NegativeInfinity))
  *      .vmap(Axis[Context])(attnScore => softmax(attnScore).relabelTo(Axis[AttnWeights]))
  *    val droppedWeights = attnWeights.vmap(Axis[Context]): w =>
  *      attentionWeightDropout.apply(w)
  *    val res = droppedWeights.dot(Axis[AttnWeights ~ Context])(values)
  *    res
  *
  * case class MultiHeadCrossAttentionHyperParams(
  *    crossAttentionWeightsDropout: DropoutLayer.HyperParams[MultiHeadCrossAttentionOld.CrossAttnWeights]
  * )
  *
  * object MultiHeadCrossAttentionOld:
  *  trait CrossAttnWeights derives Label
  *
  * case class MultiHeadCrossAttentionOld[CrossEmbedding: Label, Embedding: Label](
  *    hyperParams: MultiHeadCrossAttentionHyperParams
  * )(
  *    params: MultiHeadCrossAttentionParams[CrossEmbedding, Embedding]
  * ):
  *
  *  import MultiHeadCrossAttentionOld.CrossAttnWeights
  *
  *  private val projection = AffineLayer(params.proj)
  *  private val crossAttentionWeightsDropout = DropoutLayer(hyperParams.crossAttentionWeightsDropout)
  *
  *  def apply[CrossContext: Label, Context: Label](crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
  *    val heads = zipvmap(Axis[Head])(params.wq.weights, params.wk.weights, params.wv.weights):
  *      case (wq, wk, wv) =>
  *        attention(wq, wk, wv)(crossContext, context)
  *    heads.vmap(Axis[Context])(heads => projection(heads.flatten))
  *
  *  private def attention[CrossContext: Label, Context: Label](
  *      wq: Tensor2[Embedding, HeadQuery, Float],
  *      wk: Tensor2[CrossEmbedding, HeadKey, Float],
  *      wv: Tensor2[CrossEmbedding, HeadValue, Float]
  *  )(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, HeadValue, Float] =
  *    val queries = context.dot(Axis[Embedding])(wq)
  *    val keys = crossContext.dot(Axis[CrossEmbedding])(wk)
  *    val values = crossContext.dot(Axis[CrossEmbedding])(wv)
  *    val dk = Tensor0(Math.sqrt(keys.shape(Axis[HeadKey])).toFloat)
  *    val crossAttnScores = (queries.dot(Axis[HeadQuery ~ HeadKey])(keys) /! dk)
  *    val crossAttnWeights = crossAttnScores
  *      .vmap(Axis[Context])(attnScore => softmax(attnScore).relabelTo(Axis[CrossAttnWeights]))
  *    val droppedCrossAttnWeights = crossAttnWeights.vmap(Axis[Context])(crossAttentionWeightsDropout)
  *    val res = droppedCrossAttnWeights.dot(Axis[CrossAttnWeights ~ CrossContext])(values)
  *    res
  *
  * case class TransformerLayerOld[Context: Label, Embedding: Label](
  *    embeddingMixer: Tensor1[Embedding, Float] => Tensor1[Embedding, Float],
  *    contextMixer: Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]
  * ) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
  *
  *  def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
  *    var x = t
  *    x = x + contextMixer(x)
  *    x = x + x.vmap(Axis[Context])(embeddingMixer)
  *    x
  *
  * case class DropoutHyperParams[L](
  *    dropoutRate: Float,
  *    sourceOfRandomness: Iterator[Random.Key],
  *    isTraining: Boolean
  * ):
  *  require(0f <= dropoutRate && dropoutRate <= 1f)
  *
  * case class EmbeddingMixerHyperParams[Embedding](
  *    hiddenDropout: DropoutLayer.HyperParams[EmbeddingMixed],
  *    outputDropout: DropoutLayer.HyperParams[Embedding]
  * )
  *
  * case class MultiHeadAttentionHyperParams[Context](
  *    createAttentionMasking: AxisExtent[Context] => Tensor[(Context, Prime[Context]), Boolean],
  *    attentionDropout: DropoutLayer.HyperParams[MultiHeadAttentionOld.AttnWeights]
  * )
  *
  * object TransformerLayer:
  *  def fromParams[Context: Label, Embedding: Label](
  *      hyperParams: TransformerLayerHyperParams[Context, Embedding]
  *  )(
  *      params: TransformerLayerParams[Embedding]
  *  ): TransformerLayerOld[Context, Embedding] =
  *    val attentionPreNormalization = LayerNorm(params.attentionNorm)
  *    val embeddingMixerPreNormalization = vmap(Axis[Context])(LayerNorm(params.embeddingMixerNorm))
  *
  *    TransformerLayerOld(
  *      embeddingMixer = attentionPreNormalization andThen EmbeddingMixer(hyperParams.embeddingMixer)(params.embeddingMixer),
  *      contextMixer = embeddingMixerPreNormalization andThen MultiHeadAttentionOld(hyperParams.attn)(params.attention)
  *    )
  *
  * case class CrossTransformerLayerOld[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
  *    crossContextMixer: (Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float],
  *    transformerLayer: TransformerLayerOld[Context, Embedding]
  * ) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):
  *
  *  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
  *    transformerLayer(context + crossContextMixer(crossContext, context))
  *
  * case class CrossTransformerLayerHyperParams[CrossContext, CrossEmbedding, Context, Embedding](
  *    crossAttention: MultiHeadCrossAttention.HyperParams,
  *    transformer: TransformerLayer.HyperParams[Context, Embedding]
  * )
  *
  * object CrossTransformerLayerOld:
  *  def fromParams[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
  *      hyperParams: CrossTransformerLayerHyperParams[CrossContext, CrossEmbedding, Context, Embedding]
  *  )(
  *      params: CrossTransformerLayerParams[CrossEmbedding, Embedding]
  *  ): CrossTransformerLayerOld[CrossContext, CrossEmbedding, Context, Embedding] =
  *    val crossAttentionPreNormalization = vmap(Axis[Context])(LayerNorm(params.crossAttentionPreNormalization))
  *    val multiHeadCrossAttention = MultiHeadCrossAttentionOld(hyperParams.crossAttention)(params.crossAttention)
  *    CrossTransformerLayerOld(
  *      crossContextMixer = (crossContext, context) =>
  *        multiHeadCrossAttention(crossContext, crossAttentionPreNormalization(context)),
  *      TransformerLayer.fromParams(hyperParams.transformer)(params.transformer)
  *    )
  *
  * case class CrossTransformerBlockOld[CrossContext: Label, CrossEmbedding, Context: Label, Embedding](
  *    layers: List[CrossTransformerLayerOld[CrossContext, CrossEmbedding, Context, Embedding]]
  * ) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):
  *  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
  *    layers.foldLeft(context):
  *      case (t, layer) => layer(crossContext, context)
  *
  * case class ViTPatchingOld(params: ViTPatchingParams):
  *
  *  private val projection =
  *    val weightShape = params.projectionWeights.shape
  *    val kernelSize = (weightShape.extent(Axis[Width]), weightShape.extent(Axis[Height]))
  *    Conv2DLayer(Conv2DLayer.Params(params.projectionWeights), stride = kernelSize)
  *
  *  private def positionalEncoding2D(shape: Shape3[Width, Height, EncoderEmbedding]): Tensor3[Width, Height, EncoderEmbedding, Float] =
  *    // 1. Prepare things we need for positional encoding
  *    val widthExtent = shape.extent(Axis[Width]).size
  *    val heightExtent = shape.extent(Axis[Height]).size
  *    val embedDim = shape.extent(Axis[EncoderEmbedding]).size
  *    val scaleCount = embedDim / 4
  *    val posScales = (Tensor1(Axis[EncoderEmbedding]).fromArray(Array.range(0, scaleCount)).asFloat *! -(Tensor0(10000.0f).log / scaleCount)).exp
  *
  *    // 2. Prepare Width (X-axis)
  *    val widthPosRaw = Tensor1(Axis[Width]).fromArray(Array.range(0, widthExtent))
  *    val widthPosScaled = widthPosRaw.asFloat.vmap(Axis[Width])(_ *! posScales)
  *    val widthPosEncoded = concatenate(widthPosScaled.sin, widthPosScaled.cos, concatAxis = Axis[EncoderEmbedding])
  *
  *    // 3. Prepare Height (Y-axis)
  *    val heightPosRaw = Tensor1(Axis[Height]).fromArray(Array.range(0, heightExtent))
  *    val heightPosScaled = heightPosRaw.asFloat.vmap(Axis[Height])(_ *! posScales)
  *    val heightPosEncoded = concatenate(heightPosScaled.sin, heightPosScaled.cos, concatAxis = Axis[EncoderEmbedding])
  *
  *    // 4. Expansion into 2D Grids and Concatenation
  *    val widthPosGrid = stack(List.fill(heightExtent)(widthPosEncoded), newAxis = Axis[Height]).transpose(Axis[Width], Axis[Height], Axis[EncoderEmbedding])
  *    val heightPosGrid = stack(List.fill(widthExtent)(heightPosEncoded), newAxis = Axis[Width])
  *
  *    concatenate(widthPosGrid, heightPosGrid, concatAxis = Axis[EncoderEmbedding])
  *
  *  def apply(img: Tensor3[Width, Height, Channel, Float]): Tensor2[Width |*| Height, EncoderEmbedding, Float] =
  *    val projected = projection(img)
  *    val x = projected + positionalEncoding2D(projected.shape)
  *    x.flatten((Axis[Width], Axis[Height]))
  *
  * case class Embedder[Context: Label, Embedding: Label](vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float], positionalEmbeddings: Tensor2[Context, Embedding, Float]):
  *
  *  def apply(tokens: Tensor1[Context, Int]): Tensor2[Context, Embedding, Float] =
  *    val embeddings = vocabularyEmbeddings.take(Axis[Vocab])(tokens)
  *    embeddings + positionalEmbeddings
  */

case class Timer private (
    private val decay: Float = 0.01f
):

  private var lastTime = System.currentTimeMillis()
  private var internalRunningAverage = -1f

  def tick(): Unit =
    val now = System.currentTimeMillis()
    val elapsed = now - lastTime
    internalRunningAverage =
      if internalRunningAverage == -1f
      then elapsed
      else internalRunningAverage * decay + elapsed * (1f - decay)
    lastTime = now

  def reset(): Unit =
    lastTime = System.currentTimeMillis()
    internalRunningAverage = -1f

  def runningAvgSeconds: Float = internalRunningAverage / 1000f

object Timer:
  def start(): Timer = new Timer()
