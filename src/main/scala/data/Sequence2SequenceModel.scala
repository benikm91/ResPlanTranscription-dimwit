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
import resplan.nn.transformer.{noMask, causalMask}
import resplan.nn.VocabularyEmbedder
import resplan.nn.base.{DropoutLayer, LinearLayer}
import resplan.nn.normalization.LayerNorm
import resplan.nn.transformer.{SelfAttention, MultiHeadAttention, MultiHeadCrossAttention, CrossAttention}
import resplan.nn.transformer.{TransformerBlock, TransformerLayer, CrossTransformerLayer, CrossTransformerBlock, MLPEmbeddingMixer}
import dimwit.stats.Uniform
import RandomUtil.toSourceOfRandomness
import dimwit.hardware.DeviceBackend.GPU
import dimwit.python.PyBridge.{toPyTensor, liftPyTensor, liftPyTensor1}

import java.sql.Time
import resplan.nn.VisitionTransformer2DPatching
import dimwit.stats.Bernoulli

object Config:
  inline val numberOfEncoderLayers = 6
  inline val numberOfDecoderLayers = 6
  inline val learningRate = 1e-4f
  inline val beta1 = 0.9f
  inline val beta2 = 0.99f
  inline val weightDecayFactor = 0.01f
  inline val batchSize = 256
  inline val numberOfHeads = 6
  inline val embeddingExtent = 384
  inline val decoderContextMaxLength = 32
  inline val dropout = 0.1

  inline val validationSetSize = 512

  inline val numIterations = 1_000_000
  inline val evaluationInterval = (10_240 / batchSize).toInt

  private inline def validateConfig: Unit =
    inline if embeddingExtent % numberOfHeads != 0 then
      import scala.compiletime.{error, constValue}
      import scala.compiletime.ops.int.ToString
      scala.compiletime.error(
        "Config Error: 'embeddingExtent' must be divisible by 'numberOfHeads', but got embeddingExtent = " + constValue[ToString[embeddingExtent.type]] + " and numberOfHeads = " + constValue[ToString[numberOfHeads.type]]
      )

  validateConfig

import Config.*
import IteratorUtil.*

trait Vocab derives Label
trait EncoderEmbedding derives Label
trait DecoderEmbedding derives Label
trait EncoderContext derives Label
trait DecoderContext derives Label
trait Batch derives Label

// val nodeLinearization = RandomNodeLinearization
// val edgeLinearization = RandomEdgeLinearization
val nodeLinearization = SortedNodeLinearization
// val edgeLinearization = SortedEdgeLinearization
val edgeLinearization = NoEdgeLinearization

import resplan.nn.transformer.{Head, HeadQuery, HeadKey, HeadValue}
import resplan.nn.transformer.MLPEmbeddingMixer.EmbeddingMixed

val batchExtent = Axis[Batch] -> batchSize
val headExtent = Axis[Head] -> numberOfHeads
val headQueryExtent = Axis[HeadQuery] -> embeddingExtent / numberOfHeads
val headKeyExtent = Axis[HeadKey] -> embeddingExtent / numberOfHeads
val headValueExtent = Axis[HeadValue] -> embeddingExtent / numberOfHeads
val encoderEmbeddingExtent = Axis[EncoderEmbedding] -> embeddingExtent
val embeddingMixedExtent = Axis[EmbeddingMixed] -> 4 * encoderEmbeddingExtent.size
val decoderContextExtent = Axis[DecoderContext] -> decoderContextMaxLength
val decoderEmbeddingExtent = Axis[DecoderEmbedding] -> embeddingExtent
val patchWidthExtent = Axis[Width] -> 16
val patchHeightExtent = Axis[Height] -> 16
val channelExtent = Axis[Channel] -> 3
val vocabExtent = Axis[Vocab] -> 64

def imageNormalization(image: Tensor3[Width, Height, Channel, Int]): Tensor3[Width, Height, Channel, Float] = image.asFloat /! 255f

case class Sequence2SequenceModelParams(
    encoderLayers: List[TransformerLayer.Params[EncoderEmbedding]],
    decoderLayer: List[CrossTransformerLayer.Params[EncoderEmbedding, DecoderEmbedding]],
    vocabEmbedding: Tensor2[Vocab, DecoderEmbedding, Float],
    positionalEmbeddings: Tensor2[DecoderContext, DecoderEmbedding, Float],
    patchingParams: VisitionTransformer2DPatching.Params[Width, Height, Channel, EncoderEmbedding],
    outputLayerNormalization: LayerNorm.Params[DecoderEmbedding],
    outputProjection: LinearLayer.Params[DecoderEmbedding, Vocab]
) derives ToPyTree

object Sequence2SequenceModelParams:
  def init(key: Random.Key): Sequence2SequenceModelParams =
    val (encoderLayersKey, decoderLayersKey, vocabEmbeddingKey, positionalEmbeddingKey, vitPatchingKey, outputProjectionKey) = key.splitToTuple(6)

    val vocabScale = 1.0f / Math.sqrt(decoderEmbeddingExtent.size.toDouble).toFloat
    val posScale = 1.0f / Math.sqrt(decoderEmbeddingExtent.size.toDouble).toFloat

    Sequence2SequenceModelParams(
      encoderLayers = encoderLayersKey.split(numberOfEncoderLayers).map(key =>
        val (attnKey, mixKey) = key.split2()
        TransformerLayer.Params.defaultInit(attnKey, headExtent, headQueryExtent, headKeyExtent, headValueExtent, encoderEmbeddingExtent, embeddingMixedExtent, numberOfEncoderLayers)
      ).toList,
      decoderLayer = decoderLayersKey.split(numberOfDecoderLayers).map(key =>
        val (crossAttentionKey, transformerKey) = key.split2()
        CrossTransformerLayer.Params.defaultInit(
          crossAttentionKey,
          headExtent,
          headQueryExtent,
          headKeyExtent,
          headValueExtent,
          encoderEmbeddingExtent,
          decoderEmbeddingExtent,
          embeddingMixedExtent,
          numberOfDecoderLayers
        )
      ).toList,
      vocabEmbedding = Normal.standardIsotropic(Shape(vocabExtent, decoderEmbeddingExtent), scale = vocabScale).sample(vocabEmbeddingKey),
      positionalEmbeddings = Normal.standardIsotropic(Shape(decoderContextExtent, decoderEmbeddingExtent), scale = posScale).sample(positionalEmbeddingKey),
      patchingParams = VisitionTransformer2DPatching.Params.defaultInit(vitPatchingKey, patchWidthExtent, patchHeightExtent, channelExtent, encoderEmbeddingExtent),
      outputLayerNormalization = LayerNorm.Params.defaultInit(decoderEmbeddingExtent),
      outputProjection = LinearLayer.Params.defaultInit(decoderEmbeddingExtent, vocabExtent, outputProjectionKey)
    )

case class Sequence2SequenceModelHyperParams(
    encoderTransformerLayer: TransformerLayer.HyperParams[EncoderContext, EncoderEmbedding],
    decoderCrossTransformerLayer: CrossTransformerLayer.HyperParams[EncoderContext, DecoderContext, DecoderEmbedding]
)

case class Sequence2SequenceModelFamily(hyperParams: Sequence2SequenceModelHyperParams)(params: Sequence2SequenceModelParams):

  private val vitPatching = VisitionTransformer2DPatching(params.patchingParams)
  private val encoder = TransformerBlock(params.encoderLayers.map(TransformerLayer.WithPostNorm(hyperParams.encoderTransformerLayer)))
  private val decoder = CrossTransformerBlock(params.decoderLayer.map(CrossTransformerLayer.WithPostNorm(hyperParams.decoderCrossTransformerLayer)))
  private val embedder = VocabularyEmbedder(params.vocabEmbedding)
  private val outputLayer = LayerNorm(params.outputLayerNormalization) andThen LinearLayer(params.outputProjection)

  def logits(img: Tensor3[Width, Height, Channel, Float], shiftedTargets: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, Vocab, Float] =
    val flatPatches = vitPatching(img)
    val encoderInputContext = flatPatches.relabel(Axis[Width |*| Height].as(Axis[EncoderContext]))
    val finalEncoderContext = encoder(encoderInputContext)
    val inputContext = shiftedTargets.vmap(Axis[DecoderContext])(embedder)
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

    var finalOutputSeq = Tensor(Shape1(decoderContextExtent)).fill(paddingValue)

    // Autoregressive loop
    for i <- 0 until decoderContextExtent.size do
      val inputContext = shiftRightBOS(finalOutputSeq).vmap(Axis[DecoderContext])(embedder)
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
    val keys = key.toSourceOfRandomness
    new Sequence2SequenceModelHyperParams(
      encoderTransformerLayer = TransformerLayer.HyperParams(
        embeddingMixer = MLPEmbeddingMixer.HyperParams(gelu),
        multiHeadAttention = MultiHeadAttention.HyperParams(
          SelfAttention.HyperParams(createAttentionMask = noMask)
        )
      ),
      decoderCrossTransformerLayer = CrossTransformerLayer.HyperParams(
        multiHeadCrossAttention = MultiHeadCrossAttention.HyperParams(
          CrossAttention.HyperParams(createAttentionMask = noMask)
        ),
        embeddingMixer = MLPEmbeddingMixer.HyperParams(gelu),
        multiHeadAttention = MultiHeadAttention.HyperParams(
          SelfAttention.HyperParams(createAttentionMask = causalMask)
        )
      )
    )

val Sequence2SequenceTrainModel = Sequence2SequenceModelFamily(Sequence2SequenceModelHyperParams(Random.Key(42), true))
val Sequence2SequenceEvalModel = Sequence2SequenceModelFamily(Sequence2SequenceModelHyperParams(Random.Key(42), false))

case class BatchSample(image: Tensor[(Batch, Width, Height, Channel), Float], target: Tensor2[Batch, DecoderContext, Int]):
  def toGPU: BatchSample =
    val gpuDevice = GPU.devices.head
    BatchSample(image.toDevice(gpuDevice), target.toDevice(gpuDevice))

def shiftRight(target: Tensor1[DecoderContext, Int]): Tensor1[DecoderContext, Int] =
  val array = Jax.jnp.roll(toPyTensor(target), shift = 1, axis = -1)
  liftPyTensor1(Axis[DecoderContext], target.vtype)(array)

def shiftRightBOS(target: Tensor1[DecoderContext, Int]): Tensor1[DecoderContext, Int] =
  shiftRight(target).set(Axis[DecoderContext].at(0))(BOS)

@main def train(): Unit =
  scalaRandom.setSeed(42)
  val initialKey = Random.Key(42)
  val (trainDataKey, valDataKey, trainAugKey, initTrainKey, initModelKey) = initialKey.splitToTuple(5)
  val plans = scalaRandom.shuffle(ResPlanDataset.loadRawPlans())
  // Filter plans that do not fit in the configured context size
  val validPlans = plans.filter(plan => plan.graph.nodes.size + plan.graph.edges.size * 3 + 2 < decoderContextExtent.size)
  println(f"Filtered plans from ${plans.size} to ${validPlans.size}")
  val (trainPlans, valPlans) = validPlans.splitAt(validPlans.size - validationSetSize)
  // Load planImages as lazy np.memmap representation
  val planImages = liftPyTensor[(Data, Width, Height, Channel), Int](Jax.jnp.asarray(PythonHelper.np.memmap("res/plans/20/plan_imgs.bin", dtype = PythonHelper.np.uint8, shape = (17107, 160, 160, 3), mode = "r")))
  val trainDataset = ResPlanDataset(
    trainPlans,
    planImages,
    PaddedGraphLinearization(
      nodeLinearization,
      edgeLinearization,
      decoderContextExtent.size
    ),
    imageNormalization,
    trainDataKey.toSourceOfRandomness,
    inifinite = true
  )
  val valGraphLinearization = PaddedGraphLinearization(
    nodeLinearization,
    edgeLinearization,
    decoderContextExtent.size
  )
  val valDataset = ResPlanDataset(
    valPlans,
    planImages,
    valGraphLinearization,
    imageNormalization,
    valDataKey.toSourceOfRandomness,
    inifinite = false
  )
  val valSubset = valDataset.copy(plans = valPlans.take(32))

  def augmentImage(image: Tensor3[Width, Height, Channel, Float], key: Tensor0[Random.Key]): Tensor3[Width, Height, Channel, Float] =
    ImageAugmentation(image, key.item)

  val jitAugmentImage = jit(augmentImage)

  val trainSampleStream = trainDataset
    .iterator
    .grouped(batchSize).withPartial(false)
    .zip(trainAugKey.toSourceOfRandomness)
    .map: (sample, trainAugKey) =>
      val imageBatch = zipvmap(Axis[Batch])(
        stack(sample.map(_.image), Axis[Batch]),
        trainAugKey.splitToTensor(batchExtent)
      ): (sample, augKey) =>
        jitAugmentImage(sample, augKey)
      val targetBatch = stack(sample.map(_.lineraizedGraph), Axis[Batch])
      BatchSample(imageBatch, targetBatch).toGPU

  /*val debugTimer = Timer.start()
  for batch <- trainSampleStream do
    debugTimer.tick()
    println(f"s/batch: ${debugTimer.runningAvgSeconds}%.2f")*/
  val initParams = Sequence2SequenceModelParams.init(initModelKey)

  val adam = Adam(learningRate = learningRate, b1 = beta1, b2 = beta2, epsilon = 1e-8f)
  val adamW = AdamW(adam, weightDecayFactor = weightDecayFactor)
  type AdamWState = adamW.State[Sequence2SequenceModelParams]

  case class TrainingState(
      params: Sequence2SequenceModelParams,
      trainKey: Random.Key,
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

  def sampleSubnetwork(
      params: Sequence2SequenceModelParams,
      key: Random.Key,
      mlpDropoutRate: Float = 0.1f
  ): Sequence2SequenceModelParams =
    val (encoderKey, decoderKey) = key.splitToTuple(2)
    params.copy(
      encoderLayers = params.encoderLayers.zip(encoderKey.toSourceOfRandomness).map:
        case (params, k) =>
          val (fcKey, projKey) = k.split2()
          val keepProb = 1.0f - mlpDropoutRate
          val fcDropoutMask = IndependentDistribution.fromUnivariate(params.mlpParams.c_fc.bias.shape, Bernoulli(Prob(keepProb))).sample(fcKey).asFloat *! (1f / (keepProb))
          val projDropoutMask = IndependentDistribution.fromUnivariate(params.mlpParams.c_proj.bias.shape, Bernoulli(Prob(keepProb))).sample(projKey).asFloat *! (1f / (keepProb))
          params.copy(
            // attentionParams = ??? TODO attention dropout
            mlpParams = params.mlpParams.copy(
              c_fc = params.mlpParams.c_fc.copy(
                weight = params.mlpParams.c_fc.weight *! fcDropoutMask,
                bias = params.mlpParams.c_fc.bias * fcDropoutMask
              ),
              c_proj = params.mlpParams.c_proj.copy(
                weight = params.mlpParams.c_proj.weight *! projDropoutMask,
                bias = params.mlpParams.c_proj.bias * projDropoutMask
              )
            )
          ),
      decoderLayer = params.decoderLayer.zip(decoderKey.toSourceOfRandomness).map:
        case (params, k) => params
    )

  def gradientStep(
      imgs: Tensor[(Batch, Width, Height, Channel), Float],
      shiftedTargets: Tensor2[Batch, DecoderContext, Int],
      targets: Tensor2[Batch, DecoderContext, Int],
      state: TrainingState
  ): TrainingState =
    val (nextKey, thisKey) = state.trainKey.split2()
    val lossBatch = batchLoss(imgs, shiftedTargets, targets)
    val grads = Autodiff.grad(lossBatch)(sampleSubnetwork(state.params, thisKey))
    val loss = lossBatch(state.params) // TODO move to gradAndValue
    val (params, adamWState) = adamW.update(grads, state.params, state.adamWState)
    TrainingState(params = params, trainKey = nextKey, adamWState = adamWState, loss = loss)
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
    val logits = jitLogits(params, sample.image, shiftRightBOS(targets))
    val valPredictionTeacherForcing = logits.argmax(Axis[Vocab])
    val valLoss = loss(logits, targets)

    val valPrediction = jitGenerate(params, sample.image)
    val score = graphLinearization.score(valPrediction, targets)

    println(f"Targets: $targets")
    println(f"Predict: $valPrediction")
    println(f"Predict (TF): $valPredictionTeacherForcing")
    println(f"Score: $score")

    (valLoss, score)

  def miniBatchGradientDescent(
      samples: Iterator[BatchSample],
      startState: TrainingState
  ): Iterator[TrainingState] =
    samples.scanLeft(startState):
      case (state, sample) =>
        dimwit.gc()
        jitStep(sample.image, sample.target.vmap(Axis[Batch])(shiftRightBOS), sample.target, state)

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

  val initState = TrainingState(initParams, initTrainKey, adamW.init(initParams), Tensor0(-1f))
  val trainTrajectory = miniBatchGradientDescent(trainSampleStream, initState)
  val timer = Timer.start()
  println("Training...")
  val finalState = trainTrajectory
    .drop(1)
    .tapEvery(1):
      case (state, iter) =>
        // Training report
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
    .tapEvery(evaluationInterval):
      case (state, iter) =>
        // Evaluation Report
        println("Evaluation...")
        val scores = valSubset.iterator
          .map(_.toGPU)
          .map(sample =>
            val (valLoss, score) = evaluate(sample, state.params, valGraphLinearization)
            (valLoss.item, score)
          ).toList
        printScores(scores, iter = iter)
        timer.reset()
    .drop(numIterations - 1) // iterate to final iteration
    .next()

  val scores = valDataset.iterator
    .map(_.toGPU)
    .map(sample =>
      val (valLoss, score) = evaluate(sample, finalState.params, valGraphLinearization)
      (valLoss.item, score)
    ).toList
  println("Final scores:")
  printScores(scores)
