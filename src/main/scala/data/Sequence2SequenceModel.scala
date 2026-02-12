package resplan.data

import dimwit.*
import dimwit.Conversions.given
import nn.ActivationFunctions.*
import javax.xml.transform.Transformer
import java.util.Base64.Decoder
import nn.Adam
import nn.AdamW

object Config:
  inline val learningRate = 1e-3f
  inline val beta1 = 0.9f
  inline val beta2 = 0.99f
  inline val weightDecayFactor = 0.01f
  inline val batchSize = 32
  inline val numberOfLayers = 6
  inline val numberOfHeads = 6
  inline val extentEmbedding = 384
  inline val dropout = 0.2

  private inline def validateConfig: Unit =
    inline if extentEmbedding % numberOfHeads != 0 then
      import scala.compiletime.{error, constValue}
      import scala.compiletime.ops.int.ToString
      scala.compiletime.error(
        "Config Error: 'extentEmbedding' must be divisible by 'numberOfHeads', but got extentEmbedding = " + constValue[ToString[extentEmbedding.type]] + " and numberOfHeads = " + constValue[ToString[numberOfHeads.type]]
      )

  validateConfig

import Config.*

trait Vocab derives Label
trait Head derives Label
trait HeadKey derives Label
trait HeadQuery derives Label
trait HeadValue derives Label
trait EncoderEmbedding derives Label
trait DecoderEmbedding derives Label
trait EncoderContext derives Label
trait DecoderContext derives Label
trait EmbeddingMixed derives Label
trait Batch derives Label
import Util.given

case class Sequence2SequenceModelParams(
    encoderLayers: List[TransformerLayerParams[EncoderEmbedding]],
    decoderLayer: List[CrossTransformerLayerParams[EncoderEmbedding, DecoderEmbedding]],
    contextEmbedding: Tensor2[DecoderContext, DecoderEmbedding, Float],
    vitPatchingParams: ViTPatchingParams,
    outputLayerNormalization: LayerNormalizationParams[DecoderEmbedding],
    outputProjection: ProjectionLayerParams[DecoderEmbedding, Vocab]
) derives ToPyTree

object Sequence2SequenceModelParams:
  def init(key: Random.Key): Sequence2SequenceModelParams =
    ???

case class Sequence2SequenceModel(params: Sequence2SequenceModelParams):

  private val vitPatching = ViTPatching(params.vitPatchingParams)
  private val encoder = TransformerBlock(params.encoderLayers.map(TransformerLayer.fromParams(_, noMasking[EncoderContext])))
  private val decoder = CrossTransformerBlock(params.decoderLayer.map(CrossTransformerLayer.fromParams(Axis[EncoderContext], _, causalMasking[DecoderContext])))
  private val inputContext = params.contextEmbedding
  private val outputLayer = LayerNorm(params.outputLayerNormalization) andThen ProjectionLayer(params.outputProjection)

  def logits(img: Tensor3[Width, Height, Channel, Float]): Tensor2[DecoderContext, Vocab, Float] =
    val flatPatches = vitPatching(img)
    val encoderInputContext = flatPatches.relabel(Axis[Width |*| Height].as(Axis[EncoderContext]))
    val finalEncoderContext = encoder(encoderInputContext)
    val finalContext = decoder(finalEncoderContext, inputContext)
    finalContext.vmap(Axis[DecoderContext])(outputLayer)

  def probits(img: Tensor3[Width, Height, Channel, Float]): Tensor2[DecoderContext, Vocab, Float] =
    logits(img).vapply(Axis[Vocab])(softmax)

  def apply(img: Tensor3[Width, Height, Channel, Float]): Tensor1[DecoderContext, Int] =
    logits(img).argmax(Axis[Vocab])

@main def train(): Unit =
  println("Training not implemented yet")
  val dataset = ResPlanDataset(
    ResPlanDataset.loadRawPlans(),
    RandomGraphLinearization,
    StaticRenderer
  ).iterator.grouped(32).map: sample =>
    val imgBatch = stack(sample.map(_.image), Axis[Batch])
    // TODO pad the lineraized graphs
    val targetBatch = stack(sample.map(_.lineraizedGraph), Axis[Batch])
    (imgBatch, targetBatch)

  val initParams = Sequence2SequenceModelParams.init(Random.Key(42))

  val adam = Adam(learningRate = learningRate, b1 = beta1, b2 = beta2, epsilon = 1e-8f)
  val adamW = AdamW(adam, weightDecayFactor = weightDecayFactor)
  type AdamWState = adamW.State[Sequence2SequenceModelParams]

  case class TrainingState(
      params: Sequence2SequenceModelParams,
      adamWState: AdamWState,
      loss: Tensor0[Float]
  )

  def batchLoss(input: Tensor[(Batch, Width, Height, Channel), Float], targets: Tensor2[Batch, DecoderContext, Int])(params: Sequence2SequenceModelParams): Tensor0[Float] =
    val model = Sequence2SequenceModel(params)
    val logits = input.vmap(Axis[Batch])(model.logits)
    val lossPerSample = zipvmap(Axis[Batch])(targets, logits): (targets, logits) =>
      val lossPerContextPosition = zipvmap(Axis[DecoderContext])(targets, logits): (target, logits) =>
        Loss.crossEntropy(logits = logits, label = target)
      lossPerContextPosition.mean
    lossPerSample.mean
