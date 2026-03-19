package resplan.model

import dimwit.*
import dimwit.Conversions.given
import deepwit.*
import deepwit.labels.{Head, HeadQuery, HeadKey, HeadValue, EmbeddingMixed}
import nn.Adam
import nn.AdamW
import dimwit.stats.Normal
import scala.util.Random as scalaRandom
import scala.math.Numeric.Implicits.infixNumericOps
import scala.compiletime.ops.double
import dimwit.jax.Jax
import resplan.util.PythonSetup
import resplan.util.PythonHelper
import dimwit.stats.Uniform
import dimwit.hardware.DeviceBackend.{CPU, GPU}
import dimwit.python.PyBridge.{toPyTensor, liftPyTensor, liftPyTensor1}

import resplan.util.Timer
import resplan.util.IteratorUtil.*
import resplan.util.RandomUtil.toSourceOfRandomness

import java.sql.Time
import dimwit.stats.Bernoulli

import resplan.data.*
import scala.util.Try

object Config:
  inline val numberOfEncoderLayers = 8
  inline val numberOfDecoderLayers = 8
  inline val learningRate = 1e-4f
  inline val beta1 = 0.9f
  inline val beta2 = 0.99f
  inline val weightDecayFactor = 0.01f
  inline val batchSize = 256
  inline val numberOfHeads = 6
  inline val embeddingExtent = 384
  inline val decoderContextMaxLength = 32
  inline val dropout = 0.1f

  inline val validationSetSize = 512

  inline val numIterations = 1_000_000
  inline val trainInterval = 1
  inline val evalInterval = (10_240 / batchSize).toInt

  private inline def validateConfig: Unit =
    inline if embeddingExtent % numberOfHeads != 0 then
      import scala.compiletime.{error, constValue}
      import scala.compiletime.ops.int.ToString
      scala.compiletime.error(
        "Config Error: 'embeddingExtent' must be divisible by 'numberOfHeads', but got embeddingExtent = " + constValue[ToString[embeddingExtent.type]] + " and numberOfHeads = " + constValue[ToString[numberOfHeads.type]]
      )

  validateConfig

import Config.*

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

def imageNormalization(image: Tensor3[Width, Height, Channel, Int]): Tensor3[Width, Height, Channel, Float] =
  (image.asFloat /! 127.5f) -! 1.0f
  // (image.asFloat /! 255f)

case class Sequence2SequenceModelParams(
    encoderLayers: List[TransformerLayer.Params[EncoderEmbedding]],
    encoderFinalNorm: LayerNorm.Params[EncoderEmbedding],
    decoderLayer: List[CrossTransformerLayer.Params[EncoderEmbedding, DecoderEmbedding]],
    decoderFinalNorm: LayerNorm.Params[DecoderEmbedding],
    decoderEmbedderParams: VocabularyEmbedder.Params[Vocab, DecoderEmbedding],
    decoderPositionalInjectorParams: LearnedAbsolutePositionalInjector.Params[DecoderContext, DecoderEmbedding],
    patchingParams: ConvImageToPatchEmbedder.Params[Width, Height, Channel, EncoderEmbedding],
    outputProjection: LinearLayer.Params[DecoderEmbedding, Vocab]
) derives ToPyTree

object Sequence2SequenceModelParams:
  def init(key: Random.Key): Sequence2SequenceModelParams =
    val (encoderLayersKey, decoderLayersKey, vocabEmbeddingKey, positionalEmbeddingKey, vitPatchingKey, outputProjectionKey) = key.splitToTuple(6)

    Sequence2SequenceModelParams(
      encoderLayers = encoderLayersKey.split(numberOfEncoderLayers).map(encoderLayerKey =>
        TransformerLayer.Params.xavierUniformDepthScaled(numberOfEncoderLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, encoderEmbeddingExtent, embeddingMixedExtent, encoderLayerKey)
      ).toList,
      encoderFinalNorm = LayerNorm.Params.identity(encoderEmbeddingExtent),
      decoderLayer = decoderLayersKey.split(numberOfDecoderLayers).map(decoderLayerKey =>
        CrossTransformerLayer.Params.xavierUniformDepthScaled(numberOfDecoderLayers)(
          headExtent,
          headQueryExtent,
          headKeyExtent,
          headValueExtent,
          encoderEmbeddingExtent,
          decoderEmbeddingExtent,
          embeddingMixedExtent,
          decoderLayerKey
        )
      ).toList,
      decoderFinalNorm = LayerNorm.Params.identity(decoderEmbeddingExtent),
      decoderEmbedderParams = VocabularyEmbedder.Params.lecunUniform(vocabExtent, decoderEmbeddingExtent, vocabEmbeddingKey),
      decoderPositionalInjectorParams = LearnedAbsolutePositionalInjector.Params.lecunUniform(decoderContextExtent, decoderEmbeddingExtent, positionalEmbeddingKey),
      patchingParams = ConvImageToPatchEmbedder.Params.xavierUniform(patchWidthExtent, patchHeightExtent, channelExtent, encoderEmbeddingExtent, vitPatchingKey),
      outputProjection = LinearLayer.Params.xavierUniform(decoderEmbeddingExtent, vocabExtent, outputProjectionKey)
    )

case class Sequence2SequenceModelHyperParams(
    encoderTransformerLayer: TransformerLayer.HyperParams[EncoderContext, EncoderEmbedding],
    decoderCrossTransformerLayer: CrossTransformerLayer.HyperParams[EncoderContext, DecoderContext, DecoderEmbedding]
)

case class Sequence2SequenceModelFamily(hyperParams: Sequence2SequenceModelHyperParams)(params: Sequence2SequenceModelParams):

  private val patching = ConvImageToPatchEmbedder(params.patchingParams)
  private val encoder = Transformer(params.encoderLayers.map(TransformerLayer(hyperParams.encoderTransformerLayer)), LayerNorm(params.encoderFinalNorm))

  private val decoderEmbedder = VocabularyEmbedder(params.decoderEmbedderParams)
  private val addDecoderPositionalEncoding = LearnedAbsolutePositionalInjector(params.decoderPositionalInjectorParams)
  private val decoder = CrossTransformer(params.decoderLayer.map(CrossTransformerLayer(hyperParams.decoderCrossTransformerLayer)), LayerNorm(params.decoderFinalNorm))
  private val outputLayer = LinearLayer(params.outputProjection)

  def encode(img: Tensor3[Width, Height, Channel, Float]): Tensor2[EncoderContext, EncoderEmbedding, Float] =
    val flatPatches = patching(img)
    val inputContext = flatPatches.relabel(Axis[Width |*| Height].as(Axis[EncoderContext]))
    encoder(inputContext)

  def decode(encoderContext: Tensor2[EncoderContext, EncoderEmbedding, Float], shiftedDecoderContext: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, DecoderEmbedding, Float] =
    val inputContext = shiftedDecoderContext.vmap(Axis[DecoderContext])(decoderEmbedder)
    decoder(encoderContext, addDecoderPositionalEncoding(inputContext))

  def logits(img: Tensor3[Width, Height, Channel, Float], shiftedDecoderContext: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, Vocab, Float] =
    val encoderContext = encode(img)
    val decoderContext = decode(encoderContext, shiftedDecoderContext)
    decoderContext.vmap(Axis[DecoderContext])(outputLayer)

  def probits(img: Tensor3[Width, Height, Channel, Float], decoderContext: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, Vocab, Float] =
    logits(img, decoderContext).vapply(Axis[Vocab])(softmax)

  def apply(img: Tensor3[Width, Height, Channel, Float], decoderContext: Tensor1[DecoderContext, Int]): Tensor1[DecoderContext, Int] =
    logits(img, decoderContext).argmax(Axis[Vocab])

  def newApply(img: Tensor3[Width, Height, Channel, Float]): Graph = ???

  def generate(img: Tensor3[Width, Height, Channel, Float]): Tensor1[DecoderContext, Int] =
    val encoderContext = encode(img)

    var finalOutputSeq = Tensor(Shape1(decoderContextExtent)).fill(paddingValue)

    // Autoregressive loop
    for i <- 0 until decoderContextExtent.size do
      val shiftedDecoderContext = shiftRightBOS(finalOutputSeq)
      val decoderContext = decode(encoderContext, shiftedDecoderContext)

      // Get logits for the CURRENT position only
      val logitsAtPos = outputLayer(decoderContext.slice(Axis[DecoderContext].at(i)))
      val nextToken = logitsAtPos.argmax(Axis[Vocab])

      finalOutputSeq = finalOutputSeq.set(Axis[DecoderContext].at(i))(nextToken)

      // if nextToken == EdgeClass.EndOfEdges.id then
      //  break

    finalOutputSeq

val hyperParams = new Sequence2SequenceModelHyperParams(
  encoderTransformerLayer = TransformerLayer.HyperParams(
    embeddingMixer = MLPEmbeddingMixer.HyperParams(gelu),
    multiHeadAttention = MultiHeadSelfAttention.HyperParams(
      SelfAttention.HyperParams(createAttentionMask = identityMask)
    )
  ),
  decoderCrossTransformerLayer = CrossTransformerLayer.HyperParams(
    multiHeadCrossAttention = MultiHeadCrossAttention.HyperParams(
      CrossAttention.HyperParams(createAttentionMask = identityMask)
    ),
    embeddingMixer = MLPEmbeddingMixer.HyperParams(gelu),
    multiHeadAttention = MultiHeadSelfAttention.HyperParams(
      SelfAttention.HyperParams(createAttentionMask = causalMask)
    )
  )
)
val Sequence2SequenceModel = Sequence2SequenceModelFamily(hyperParams)

case class BatchSample(image: Tensor[(Batch, Width, Height, Channel), Float], target: Tensor2[Batch, DecoderContext, Int]):
  def toGPU: BatchSample =
    val gpuDevice = Try(GPU.devices.head).getOrElse:
      println("WARNING: No GPU found, using CPU instead.")
      CPU.devices.headOption.getOrElse(throw new RuntimeException("No CPU found."))
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

  def augmentImage(image: Tensor3[Width, Height, Channel, Float], key: Tensor0[Random.Key]): Tensor3[Width, Height, Channel, Float] = ImageAugmentation(image, key.item)

  val jitAugmentImage = jit(augmentImage)

  val trainSampleStream = trainDataset
    .iterator
    .grouped(batchSize).withPartial(false)
    .zip(trainAugKey.toSourceOfRandomness).map: (sample, trainAugKey) =>
      val imageBatch = stack(sample.map(_.image), Axis[Batch])
      val targetBatch = stack(sample.map(_.lineraizedGraph), Axis[Batch])
      val augmentedImageBatch = zipvmap(Axis[Batch])(imageBatch, trainAugKey.splitToTensor(batchExtent))(jitAugmentImage.tupled)
      BatchSample(augmentedImageBatch, targetBatch).toGPU

  /*val debugTimer = Timer.start()
  for batch <- trainSampleStream do
    debugTimer.tick()
    println(f"s/batch: ${debugTimer.runningAvgSeconds}%.2f")*/
  val initParams = Sequence2SequenceModelParams.init(initModelKey)

  val adamW = AdamW(
    Adam(learningRate = learningRate, b1 = beta1, b2 = beta2),
    weightDecayFactor = weightDecayFactor
  )

  case class TrainingState(
      step: Tensor0[Int],
      params: Sequence2SequenceModelParams,
      trainKey: Random.Key,
      adamWState: adamW.State[Sequence2SequenceModelParams],
      lastStepCost: Tensor0[Float]
  )

  def loss(logits: Tensor2[DecoderContext, Vocab, Float], targets: Tensor1[DecoderContext, Int]): Tensor0[Float] =
    val lossPerContextPosition = zipvmap(Axis[DecoderContext])(targets, logits): (target, logits) =>
      crossEntropy(logits = logits, label = target)
    // mask out padding tokens for loss
    val paddingMask = !targets.elementEquals(Tensor(targets.shape).fill(paddingValue))
    val zeros = Tensor(paddingMask.shape).fill(0f)
    where(paddingMask, lossPerContextPosition, zeros).sum / paddingMask.asFloat.sum

  def costFnFor[Sample: Label](
      imgs: Tensor[(Sample, Width, Height, Channel), Float],
      targets: Tensor2[Sample, DecoderContext, Int]
  )(
      params: Sequence2SequenceModelParams
  ): Tensor0[Float] =
    val model = Sequence2SequenceModel(params)
    val shiftedTargets = targets.vmap(Axis[Sample])(shiftRightBOS)
    val logits = zipvmap(Axis[Sample])(imgs, shiftedTargets)(model.logits)
    val sampleLosses = zipvmap(Axis[Sample])(logits, targets)(loss)
    sampleLosses.mean

  def sampleThinSubnetwork(
      params: Sequence2SequenceModelParams,
      key: Random.Key
  ): Sequence2SequenceModelParams =

    val (embKey, posEmbKey, encoderKey, decoderKey) = key.splitToTuple(4)
    params.copy(
      decoderEmbedderParams = sampleThinVocabularyEmbedder(params.decoderEmbedderParams, dropout, embKey),
      decoderPositionalInjectorParams = sampleThinLearnedAbsolutePositionalInjector(params.decoderPositionalInjectorParams, dropout, posEmbKey),
      encoderLayers = params.encoderLayers.zip(encoderKey.toSourceOfRandomness).map:
        case (params, k) =>
          val (expandKey, projectKey, attentionKey) = k.splitToTuple(3)
          params.copy(
            attentionParams = params.attentionParams.copy(
              wv = zipvmap(Axis[Head])(params.attentionParams.wv, attentionKey.splitToTensor(headExtent)): (v, key) =>
                sampleThinProjection(v, dropout, key.item)
            ),
            mlpParams = params.mlpParams.copy(
              expand = sampleThinAffineLayer(params.mlpParams.expand, dropout, expandKey),
              project = sampleThinAffineLayer(params.mlpParams.project, dropout, projectKey)
            )
          ),
      decoderLayer = params.decoderLayer.zip(decoderKey.toSourceOfRandomness).map:
        case (params, k) =>
          val (expandKey, projectKey, selfAttentionKey, crossAttentionKey) = k.splitToTuple(4)
          params.copy(
            selfAttentionParams = params.selfAttentionParams.copy(
              wv = zipvmap(Axis[Head])(params.selfAttentionParams.wv, selfAttentionKey.splitToTensor(headExtent)): (v, key) =>
                sampleThinProjection(v, dropout, key.item)
            ),
            crossAttentionParams = params.crossAttentionParams.copy(
              wv = zipvmap(Axis[Head])(params.crossAttentionParams.wv, crossAttentionKey.splitToTensor(headExtent)): (v, key) =>
                sampleThinProjection(v, dropout, key.item)
            ),
            mlpParams = params.mlpParams.copy(
              expand = sampleThinAffineLayer(params.mlpParams.expand, dropout, expandKey),
              project = sampleThinAffineLayer(params.mlpParams.project, dropout, projectKey)
            )
          )
    )

  def batchGradientDescentStep(
      sample: BatchSample,
      state: TrainingState
  ): TrainingState =
    val (nextKey, thisKey) = state.trainKey.splitToTuple(2)
    val costFn = costFnFor(sample.image, sample.target)
    val grads = Autodiff.grad(costFn)(sampleThinSubnetwork(state.params, thisKey))
    val stepCost = costFn(state.params) // TODO move to gradAndValue
    val (params, adamWState) = adamW.update(grads, state.params, state.adamWState)
    TrainingState(step = state.step + 1, params = params, trainKey = nextKey, adamWState = adamWState, lastStepCost = stepCost)
  val jitBatchGradientDescentStep = jitDonatingUnsafe(batchGradientDescentStep)

  def predLogits(params: Sequence2SequenceModelParams, image: Tensor[(Width, Height, Channel), Float], shiftedTargets: Tensor1[DecoderContext, Int]): Tensor2[DecoderContext, Vocab, Float] =
    Sequence2SequenceModel(params).logits(image, shiftedTargets)
  val jitPredLogits = eagerCleanup(predLogits)

  def generate(params: Sequence2SequenceModelParams, img: Tensor3[Width, Height, Channel, Float]): Tensor1[DecoderContext, Int] =
    Sequence2SequenceModel(params).generate(img)

  val jitGenerate = jit(generate)

  def evaluate(
      sample: Sample,
      params: Sequence2SequenceModelParams,
      graphLinearization: GraphLinearization
  ): (Tensor0[Float], GraphLinearizationScore) =
    val targets = sample.lineraizedGraph
    val logits = jitPredLogits(params, sample.image, shiftRightBOS(targets))
    val valPredictionTeacherForcing = logits.argmax(Axis[Vocab])
    val valLoss = loss(logits, targets)

    val valPrediction = jitGenerate(params, sample.image)
    val score = graphLinearization.score(valPrediction, targets)

    println(f"Targets: $targets")
    println(f"Predict: $valPrediction")
    println(f"Pred_TF: $valPredictionTeacherForcing")
    println(f"Score: $score")

    (valLoss, score)

  def batchGradientDescent(
      samples: Iterator[BatchSample],
      startState: TrainingState
  ): Iterator[TrainingState] =
    samples.scanLeft(startState):
      case (state, sample) =>
        dimwit.gc()
        jitBatchGradientDescentStep(sample, state)

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

  val initState = TrainingState(0, initParams, initTrainKey, adamW.init(initParams), Tensor0(Float.NaN))
  val trainTrajectory = batchGradientDescent(trainSampleStream, initState)
  val timer = Timer.start()
  println("Training...")
  val finalState = trainTrajectory
    .drop(1)
    .tapEvery(trainInterval):
      case (state, iter) =>
        // Training report
        timer.tick()
        val secondsPerBatch = timer.runningAvgSeconds
        println(
          List(
            s"iter $iter",
            f"samples/s: ${batchSize / (secondsPerBatch)}%.2f",
            f"s/batch: $secondsPerBatch%.2f",
            f"loss: ${state.lastStepCost.item}%.2f"
          ).mkString(", ")
        )
    .tapEvery(evalInterval):
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
    .nextAfter(numIterations)

  val scores = valDataset.iterator
    .map(_.toGPU)
    .map(sample =>
      val (valLoss, score) = evaluate(sample, finalState.params, valGraphLinearization)
      (valLoss.item, score)
    ).toList
  println("Final scores:")
  printScores(scores)
