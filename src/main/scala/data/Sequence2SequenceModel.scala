package resplan.data

import dimwit.*
import dimwit.Conversions.given
import nn.ActivationFunctions.*
import javax.xml.transform.Transformer
import java.util.Base64.Decoder
import nn.Adam
import nn.AdamW
import dimwit.stats.Normal
import scala.util.Random as scalaRandom
import scala.math.Numeric.Implicits.infixNumericOps
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import scala.compiletime.ops.double
import dimwit.jax.Jax
import resplan.util.PythonSetup
import resplan.util.PythonHelper

object Config:
  inline val learningRate = 1e-4f
  inline val beta1 = 0.9f
  inline val beta2 = 0.99f
  inline val weightDecayFactor = 0.01f
  inline val batchSize = 256
  inline val numberOfLayers = 6
  inline val numberOfHeads = 6
  inline val extentEmbedding = 384
  inline val dropout = 0.2

  inline val numIterations = 1_000_000
  inline val evaluationInterval = (10_240 / batchSize).toInt

  private inline def validateConfig: Unit =
    inline if extentEmbedding % numberOfHeads != 0 then
      import scala.compiletime.{error, constValue}
      import scala.compiletime.ops.int.ToString
      scala.compiletime.error(
        "Config Error: 'extentEmbedding' must be divisible by 'numberOfHeads', but got extentEmbedding = " + constValue[ToString[extentEmbedding.type]] + " and numberOfHeads = " + constValue[ToString[numberOfHeads.type]]
      )

  validateConfig

import Config.*
import IteratorUtil.*

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
import resplan.data as Context

val numberOfEncoderLayers = 4
val numberOfDecoderLayers = 4
val headExtent = Axis[Head] -> 6
val headQueryExtent = Axis[HeadQuery] -> 64
val headKeyExtent = Axis[HeadKey] -> 64
val headValueExtent = Axis[HeadValue] -> 64
val encoderEmbeddingExtent = Axis[EncoderEmbedding] -> 384
val embeddingMixedExtent = Axis[EmbeddingMixed] -> 4 * encoderEmbeddingExtent.size
val decoderContextExtent = Axis[DecoderContext] -> 64
val decoderEmbeddingExtent = Axis[DecoderEmbedding] -> 384
val patchWidthExtent = Axis[Width] -> 16
val patchHeightExtent = Axis[Height] -> 16
val channelExtent = Axis[Channel] -> 3
val vocabExtent = Axis[Vocab] -> 64

def imageNormalization(image: Tensor3[Width, Height, Channel, Int]): Tensor3[Width, Height, Channel, Float] = image.asFloat /! 255f

case class Sequence2SequenceModelParams(
    encoderLayers: List[TransformerLayerParams[EncoderEmbedding]],
    decoderLayer: List[CrossTransformerLayerParams[EncoderEmbedding, DecoderEmbedding]],
    vocabEmbedding: Tensor2[Vocab, DecoderEmbedding, Float],
    positionalEmbeddings: Tensor2[DecoderContext, DecoderEmbedding, Float],
    vitPatchingParams: ViTPatchingParams,
    outputLayerNormalization: LayerNormalizationParams[DecoderEmbedding],
    outputProjection: ProjectionLayerParams[DecoderEmbedding, Vocab]
)

object Sequence2SequenceModelParams:
  def init(key: Random.Key): Sequence2SequenceModelParams =
    val keys = key.split(6)
    val encoderLayersKey = keys(0)
    val decoderLayersKey = keys(1)
    val vocabEmbeddingKey = keys(2)
    val positionalEmbeddingKey = keys(3)
    val vitPatchingKey = keys(4)
    val outputProjectionKey = keys(5)
    Sequence2SequenceModelParams(
      encoderLayers = encoderLayersKey.split(numberOfEncoderLayers).map(key =>
        val (attnKey, mixKey) = key.split2()
        TransformerLayerParams.init(attnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, encoderEmbeddingExtent, embeddingMixedExtent)
      ).toList,
      decoderLayer = decoderLayersKey.split(numberOfDecoderLayers).map(key =>
        val (crossAttentionKey, transformerKey) = key.split2()
        CrossTransformerLayerParams(
          crossAttentionPreNormalization = LayerNormalizationParams.init(decoderEmbeddingExtent),
          crossAttention = MultiHeadCrossAttentionParams.init(crossAttentionKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, encoderEmbeddingExtent, decoderEmbeddingExtent),
          transformer = TransformerLayerParams.init(transformerKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, decoderEmbeddingExtent, embeddingMixedExtent)
        )
      ).toList,
      vocabEmbedding = Normal.standardIsotropic(Shape(vocabExtent, decoderEmbeddingExtent), scale = 0.02f).sample(vocabEmbeddingKey),
      positionalEmbeddings = Normal.standardIsotropic(Shape(decoderContextExtent, decoderEmbeddingExtent), scale = 0.02f).sample(positionalEmbeddingKey),
      vitPatchingParams = ViTPatchingParams.init(vitPatchingKey, patchWidthExtent, patchHeightExtent, channelExtent, encoderEmbeddingExtent),
      outputLayerNormalization = LayerNormalizationParams.init(decoderEmbeddingExtent),
      outputProjection = ProjectionLayerParams.init(outputProjectionKey, decoderEmbeddingExtent, vocabExtent)
    )

case class Sequence2SequenceModelHyperParams(
    encoderTransformerLayer: TransformerLayerHyperParams[EncoderContext, EncoderEmbedding],
    decoderCrossTransformerLayer: CrossTransformerLayerHyperParams[EncoderContext, EncoderEmbedding, DecoderContext, DecoderEmbedding]
)

case class Sequence2SequenceModelFamily(hyperParams: Sequence2SequenceModelHyperParams)(params: Sequence2SequenceModelParams):

  private val vitPatching = ViTPatching(params.vitPatchingParams)
  private val encoder = TransformerBlock(params.encoderLayers.map(TransformerLayer.fromParams(hyperParams.encoderTransformerLayer)))
  private val decoder = CrossTransformerBlock(params.decoderLayer.map(CrossTransformerLayer.fromParams(hyperParams.decoderCrossTransformerLayer)))
  private val embedder = Embedder(params.vocabEmbedding, params.positionalEmbeddings)
  private val outputLayer = LayerNorm(params.outputLayerNormalization) andThen ProjectionLayer(params.outputProjection)

  def logits(img: Tensor3[Width, Height, Channel, Float], shiftedTargets: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, Vocab, Float] =
    val flatPatches = vitPatching(img)
    val encoderInputContext = flatPatches.relabel(Axis[Width |*| Height].as(Axis[EncoderContext]))
    val finalEncoderContext = encoder(encoderInputContext)
    val inputContext = embedder(shiftedTargets)
    val finalContext = decoder(finalEncoderContext, inputContext)
    finalContext.vmap(Axis[DecoderContext])(outputLayer)

  def probits(img: Tensor3[Width, Height, Channel, Float], shiftedTargets: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, Vocab, Float] =
    logits(img, shiftedTargets).vapply(Axis[Vocab])(softmax)

  def apply(img: Tensor3[Width, Height, Channel, Float], shiftedTargets: Tensor1[DecoderContext, Int]): Tensor1[DecoderContext, Int] =
    logits(img, shiftedTargets).argmax(Axis[Vocab])

  def generate(img: Tensor3[Width, Height, Channel, Float]): Tensor1[DecoderContext, Int] =
    val flatPatches = vitPatching(img)
    val encoderInputContext = flatPatches.relabel(Axis[Width |*| Height].as(Axis[EncoderContext]))
    val finalEncoderContext = encoder(encoderInputContext)

    // Initialize the sequence with BOS (Beginning of Sequence) and Padding
    var finalOutputSeq = Tensor(Shape1(decoderContextExtent)).fill(paddingValue)

    // Autoregressive loop

    for i <- 0 until decoderContextExtent.size do
      val currentInputSeq = shiftRight(finalOutputSeq).set(Axis[DecoderContext].at(0))(BOS)
      val inputContext = embedder(currentInputSeq)
      val finalContext = decoder(finalEncoderContext, inputContext)

      // Get logits for the CURRENT position only
      val logitsAtPos = outputLayer(finalContext.slice(Axis[DecoderContext].at(i)))
      val nextToken = logitsAtPos.argmax(Axis[Vocab])

      finalOutputSeq = finalOutputSeq.set(Axis[DecoderContext].at(i))(nextToken)

      // if nextToken == EdgeClass.EndOfEdges.id then
      //  break

    finalOutputSeq

object Sequence2SequenceModelHyperParams:
  def apply(key: Random.Key, isTraining: Boolean): Sequence2SequenceModelHyperParams =
    val keys = key.split(7).iterator
    new Sequence2SequenceModelHyperParams(
      encoderTransformerLayer = TransformerLayerHyperParams(
        embeddingMixer = EmbeddingMixerHyperParams(
          hiddenDropout = DropoutHyperParams(0.1f, keys.next(), isTraining),
          outputDropout = DropoutHyperParams(0.1f, keys.next(), isTraining)
        ),
        attn = MultiHeadAttentionHyperParams(
          attentionMasking = noMasking,
          attentionDropout = DropoutHyperParams(0.1f, keys.next(), isTraining)
        )
      ),
      decoderCrossTransformerLayer = CrossTransformerLayerHyperParams(
        crossAttention = MultiHeadCrossAttentionHyperParams(
          DropoutHyperParams(0.1f, keys.next(), isTraining)
        ),
        transformer = TransformerLayerHyperParams(
          embeddingMixer = EmbeddingMixerHyperParams(
            hiddenDropout = DropoutHyperParams(0.1f, keys.next(), isTraining),
            outputDropout = DropoutHyperParams(0.1f, keys.next(), isTraining)
          ),
          attn = MultiHeadAttentionHyperParams(
            attentionMasking = causalMasking,
            attentionDropout = DropoutHyperParams(0.1f, keys.next(), isTraining)
          )
        )
      )
    )

val Sequence2SequenceTrainModel = Sequence2SequenceModelFamily(Sequence2SequenceModelHyperParams(Random.Key(42), true))
val Sequence2SequenceEvalModel = Sequence2SequenceModelFamily(Sequence2SequenceModelHyperParams(Random.Key(42), false))

case class BatchSample(image: Tensor[(Batch, Width, Height, Channel), Float], target: Tensor2[Batch, DecoderContext, Int])

def shiftRight(target: Tensor1[DecoderContext, Int]): Tensor1[DecoderContext, Int] =
  val array = Jax.jnp.roll(target.jaxValue, shift = 1, axis = -1) // TODO add to dimwit
  array.at.bracketAccess(0).set(BOS)
  Tensor1.fromPy(Axis[DecoderContext], target.vtype)(array)

@main def train(): Unit =
  scalaRandom.setSeed(42)
  val plans = scalaRandom.shuffle(ResPlanDataset.loadRawPlans())
  // Filter plans that do not fit in the configured context size
  val validPlans = plans.filter(plan => plan.graph.nodes.size + plan.graph.edges.size * 3 + 2 < decoderContextExtent.size)
  println(f"Filtered plans from ${plans.size} to ${validPlans.size}")
  val (trainPlans, valPlans) = validPlans.splitAt(12800)
  // Load planImages as lazy np.memmap representation
  val planImages = Tensor.fromPy[(Data, Width, Height, Channel), Int](VType[Int])(Jax.jnp.asarray(PythonHelper.np.memmap("res/plans/20/plan_imgs.bin", dtype = PythonHelper.np.uint8, shape = (17107, 160, 160, 3), mode = "r")))
  val trainDataset = ResPlanDataset(
    trainPlans,
    planImages,
    RandomGraphLinearization(decoderContextExtent.size),
    imageNormalization,
    inifinite = true
  )
  val valGraphLinearization = RandomGraphLinearization(decoderContextExtent.size)
  val valDataset = ResPlanDataset(
    valPlans,
    planImages,
    valGraphLinearization,
    imageNormalization,
    inifinite = false
  )
  val valSubset = valDataset.copy(plans = valPlans.take(32))
  val trainSampleStream = trainDataset
    .iterator
    .grouped(batchSize).withPartial(false)
    .map: sample =>
      val imgBatch = stack(sample.map(_.image), Axis[Batch])
      val targetBatch = stack(sample.map(_.lineraizedGraph), Axis[Batch])
      BatchSample(imgBatch, targetBatch)

  val initParams = Sequence2SequenceModelParams.init(Random.Key(42))

  val adam = Adam(learningRate = learningRate, b1 = beta1, b2 = beta2, epsilon = 1e-8f)
  val adamW = AdamW(adam, weightDecayFactor = weightDecayFactor)
  type AdamWState = adamW.State[Sequence2SequenceModelParams]

  case class TrainingState(
      params: Sequence2SequenceModelParams,
      adamWState: AdamWState,
      loss: Tensor0[Float]
  )

  def loss(logits: Tensor2[DecoderContext, Vocab, Float], targets: Tensor1[DecoderContext, Int]): Tensor0[Float] =
    val lossPerContextPosition = zipvmap(Axis[DecoderContext])(targets, logits): (target, logits) =>
      Loss.crossEntropy(logits = logits, label = target)
    // mask out padding tokens for loss
    val paddingMask = !targets.elementEquals(Tensor(targets.shape).fill(paddingValue))
    val zeros = Tensor(paddingMask.shape).fill(0f)
    where(paddingMask, lossPerContextPosition, zeros).sum / paddingMask.asFloat.sum

  def batchLoss(imgs: Tensor[(Batch, Width, Height, Channel), Float], shiftedTargets: Tensor2[Batch, DecoderContext, Int], targets: Tensor2[Batch, DecoderContext, Int])(params: Sequence2SequenceModelParams): Tensor0[Float] =
    val model = Sequence2SequenceTrainModel(params)
    val logits = zipvmap(Axis[Batch])(imgs, shiftedTargets)(model.logits)
    val batchLosses = zipvmap(Axis[Batch])(logits, targets)(loss)
    batchLosses.mean

  def gradientStep(
      imgs: Tensor[(Batch, Width, Height, Channel), Float],
      shiftedTargets: Tensor2[Batch, DecoderContext, Int],
      targets: Tensor2[Batch, DecoderContext, Int],
      state: TrainingState
  ): TrainingState =
    val lossBatch = batchLoss(imgs, shiftedTargets, targets)
    val grads = Autodiff.grad(lossBatch)(state.params)
    val loss = lossBatch(state.params) // TODO move to gradAndValue
    val (params, adamWState) = adamW.update(grads, state.params, state.adamWState)
    TrainingState(params = params, adamWState = adamWState, loss = loss)
  val jitStep = jitDonatingUnsafe(gradientStep)

  def calcLogits(params: Sequence2SequenceModelParams, image: Tensor[(Width, Height, Channel), Float], shiftedTargets: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, Vocab, Float] =
    Sequence2SequenceEvalModel(params).logits(image, shiftedTargets)
  val jitLogits = eagerCleanup(calcLogits)

  def generate(params: Sequence2SequenceModelParams, img: Tensor3[Width, Height, Channel, Float]): Tensor1[DecoderContext, Int] =
    val model = Sequence2SequenceEvalModel(params)
    model.generate(img)

  val jitGenerate = jit(generate)

  def evaluate(
      sample: Sample,
      params: Sequence2SequenceModelParams,
      graphLinearization: GraphLinearization
  ): (Tensor0[Float], GraphLinearizationScore) =
    val targets = sample.lineraizedGraph
    val logits = jitLogits(params, sample.image, shiftRight(targets))
    val valLoss = loss(logits, targets)

    val valPrediction = jitGenerate(params, sample.image)
    val score = graphLinearization.score(valPrediction, targets)

    println(f"Targets: $targets")
    println(f"Predict: $valPrediction")
    println(f"Score: $score")

    (valLoss, score)

  def miniBatchGradientDescent(
      samples: Iterator[BatchSample],
      startState: TrainingState
  ): Iterator[TrainingState] =
    samples.scanLeft(startState):
      case (state, sample) =>
        dimwit.gc()
        jitStep(sample.image, sample.target.vmap(Axis[Batch])(shiftRight), sample.target, state)

  def printScores(scores: List[(Float, GraphLinearizationScore)], iter: Int = -1): Unit =
    val macroValLoss = scores.map(_._1).sum / scores.size
    val accuracyStructureCorrect = scores.map(_._2.structureCorrect.toFloat).sum / scores.size
    val accuracyNodesCorrect = scores.map(_._2.nodesCorrect.toFloat).sum / scores.size
    val accuracyGraphCorrect = scores.map(_._2.graphCorrect.toFloat).sum / scores.size
    println(
      List(
        f"iter: $iter",
        f"macro val loss: ${macroValLoss}%.2f",
        f"acc struc: ${accuracyStructureCorrect * 100}%.1f %%",
        f"acc nodes: ${accuracyNodesCorrect * 100}%.1f %%",
        f"acc graph: ${accuracyGraphCorrect * 100}%.1f %%"
      ).mkString(", ")
    )

  val initState = TrainingState(initParams, adamW.init(initParams), Tensor0(-1f))
  val trainTrajectory = miniBatchGradientDescent(trainSampleStream, initState)
  val timer = Timer.start()
  println("Training...")
  val finalState = trainTrajectory.zipWithIndex
    .drop(1)
    .tapEach:
      case (state, iter) =>
        timer.tick()
        val secondsPerBatch = timer.runningAvgSeconds
        println(
          List(
            s"iter $iter",
            f"samples/s: ${batchSize / (secondsPerBatch)}%.2f",
            f"s/batch: $secondsPerBatch%.2f",
            f"loss: ${state.loss.item}%.2f"
          ).mkString(", ")
        )
    .tapEach:
      case (state, iter) =>
        if iter % evaluationInterval == 0 then
          println("Evaluation...")
          val scores = valSubset.iterator.map(sample =>
            val (valLoss, score) = evaluate(sample, state.params, valGraphLinearization)
            (valLoss.item, score)
          ).toList
          printScores(scores, iter = iter)
          timer.reset()
    .drop(numIterations - 1) // iterate to final iteration
    .map(t => t._1)
    .next()

  val scores = valDataset.iterator.map(sample =>
    val (valLoss, score) = evaluate(sample, finalState.params, valGraphLinearization)
    (valLoss.item, score)
  ).toList
  println("Final scores:")
  printScores(scores)
