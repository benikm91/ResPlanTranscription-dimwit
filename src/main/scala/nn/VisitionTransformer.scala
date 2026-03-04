package resplan.nn

import dimwit.*
import dimwit.Conversions.given
import nn.Conv2DLayer
import dimwit.stats.Normal

trait IVisitionTransformer2DPatching[
    Width: Label,
    Height: Label,
    Channel: Label,
    PatchEmbedding: Label
] extends (Tensor3[Width, Height, Channel, Float] => Tensor2[Width |*| Height, PatchEmbedding, Float]):

  def encodeToPatches(img: Tensor[(Width, Height, Channel), Float]): Tensor[(Width, Height, PatchEmbedding), Float]
  def add2DPositionalEncoding(patches: Tensor[(Width, Height, PatchEmbedding), Float]): Tensor[(Width, Height, PatchEmbedding), Float]

  override def apply(img: Tensor3[Width, Height, Channel, Float]): Tensor2[Width |*| Height, PatchEmbedding, Float] =
    val patches = add2DPositionalEncoding(encodeToPatches(img))
    patches.flatten((Axis[Width], Axis[Height]))

case class VisitionTransformer2DPatching[
    Width: Label,
    Height: Label,
    Channel: Label,
    PatchEmbedding: Label
](
    params: VisitionTransformer2DPatching.Params[Width, Height, Channel, PatchEmbedding]
) extends IVisitionTransformer2DPatching[Width, Height, Channel, PatchEmbedding]:

  override def encodeToPatches(img: Tensor[(Width, Height, Channel), Float]): Tensor[(Width, Height, PatchEmbedding), Float] =
    val weightShape = params.projectionWeights.shape
    val kernelSize = (weightShape.extent(Axis[Width]), weightShape.extent(Axis[Height]))
    Conv2DLayer(Conv2DLayer.Params(params.projectionWeights), stride = kernelSize)(img)

  def add2DPositionalEncoding(patches: Tensor[(Width, Height, PatchEmbedding), Float]): Tensor[(Width, Height, PatchEmbedding), Float] =
    positionalEncoding2D(patches.shape) + patches

  private def positionalEncoding2D[Width: Label, Height: Label, Embedding: Label](shape: Shape3[Width, Height, Embedding]): Tensor3[Width, Height, Embedding, Float] =
    // 1. Prepare things we need for positional encoding
    val widthExtent = shape.extent(Axis[Width]).size
    val heightExtent = shape.extent(Axis[Height]).size
    val embedDim = shape.extent(Axis[Embedding]).size
    val scaleCount = embedDim / 4
    val posScales = (Tensor1(Axis[Embedding]).fromArray(Array.range(0, scaleCount)).asFloat *! -(Tensor0(10000.0f).log / scaleCount)).exp

    // 2. Prepare Width (X-axis)
    val widthPosRaw = Tensor1(Axis[Width]).fromArray(Array.range(0, widthExtent))
    val widthPosScaled = widthPosRaw.asFloat.vmap(Axis[Width])(_ *! posScales)
    val widthPosEncoded = concatenate(widthPosScaled.sin, widthPosScaled.cos, concatAxis = Axis[Embedding])

    // 3. Prepare Height (Y-axis)
    val heightPosRaw = Tensor1(Axis[Height]).fromArray(Array.range(0, heightExtent))
    val heightPosScaled = heightPosRaw.asFloat.vmap(Axis[Height])(_ *! posScales)
    val heightPosEncoded = concatenate(heightPosScaled.sin, heightPosScaled.cos, concatAxis = Axis[Embedding])

    // 4. Expansion into 2D Grids and Concatenation
    val widthPosGrid = stack(List.fill(heightExtent)(widthPosEncoded), newAxis = Axis[Height]).transpose(Axis[Width], Axis[Height], Axis[Embedding])
    val heightPosGrid = stack(List.fill(widthExtent)(heightPosEncoded), newAxis = Axis[Width])

    concatenate(widthPosGrid, heightPosGrid, concatAxis = Axis[Embedding])

object VisitionTransformer2DPatching:

  case class Params[PatchWidth, PatchHeight, Channel, PatchEmbedding](
      projectionWeights: Tensor[(PatchWidth, PatchHeight, Channel, PatchEmbedding), Float]
  )

  object Params:
    def defaultInit[PatchWidth: Label, PatchHeight: Label, Channel: Label, PatchEmbedding: Label](key: Random.Key, patchWidthExtent: AxisExtent[PatchWidth], patchHeightExtent: AxisExtent[PatchHeight], channelExtent: AxisExtent[Channel], embeddingExtent: AxisExtent[PatchEmbedding]): Params[PatchWidth, PatchHeight, Channel, PatchEmbedding] =
      Params(
        projectionWeights = Normal.standardIsotropic(Shape(patchWidthExtent, patchHeightExtent, channelExtent, embeddingExtent), scale = 0.02f).sample(key)
      )
