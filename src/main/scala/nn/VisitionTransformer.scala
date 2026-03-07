package resplan.nn

import dimwit.*
import dimwit.Conversions.given
import nn.Conv2DLayer
import dimwit.stats.Normal
import resplan.nn.init

trait IVisitionTransformer2DPatching[
    Width: Label,
    Height: Label,
    Channel: Label,
    PatchEmbedding: Label
] extends (Tensor3[Width, Height, Channel, Float] => Tensor2[Width |*| Height, PatchEmbedding, Float]):

  def encodeToPatches(img: Tensor[(Width, Height, Channel), Float]): Tensor3[Width, Height, PatchEmbedding, Float]
  def positionalEncoding2D(patches: Shape3[Width, Height, PatchEmbedding]): Tensor3[Width, Height, PatchEmbedding, Float]

  override def apply(img: Tensor3[Width, Height, Channel, Float]): Tensor2[Width |*| Height, PatchEmbedding, Float] =
    val patches = encodeToPatches(img)
    val patchesPos = patches + positionalEncoding2D(patches.shape)
    patchesPos.flatten((Axis[Width], Axis[Height]))

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

  override def positionalEncoding2D(shape: Shape3[Width, Height, PatchEmbedding]): Tensor3[Width, Height, PatchEmbedding, Float] =
    // 1. Prepare things we need for positional encoding
    val widthExtent = shape.extent(Axis[Width]).size
    val heightExtent = shape.extent(Axis[Height]).size
    val embedDim = shape.extent(Axis[PatchEmbedding]).size
    val scaleCount = embedDim / 4
    val posScales = (Tensor1(Axis[PatchEmbedding]).fromArray(Array.range(0, scaleCount)).asFloat *! -(Tensor0(10000.0f).log / scaleCount)).exp

    // 2. Prepare Width (X-axis)
    val widthPosRaw = Tensor1(Axis[Width]).fromArray(Array.range(0, widthExtent))
    val widthPosScaled = widthPosRaw.asFloat.vmap(Axis[Width])(_ *! posScales)
    val widthPosEncoded = concatenate(widthPosScaled.sin, widthPosScaled.cos, concatAxis = Axis[PatchEmbedding])

    // 3. Prepare Height (Y-axis)
    val heightPosRaw = Tensor1(Axis[Height]).fromArray(Array.range(0, heightExtent))
    val heightPosScaled = heightPosRaw.asFloat.vmap(Axis[Height])(_ *! posScales)
    val heightPosEncoded = concatenate(heightPosScaled.sin, heightPosScaled.cos, concatAxis = Axis[PatchEmbedding])

    // 4. Expansion into 2D Grids and Concatenation
    val widthPosGrid = stack(List.fill(heightExtent)(widthPosEncoded), newAxis = Axis[Height]).transpose(Axis[Width], Axis[Height], Axis[PatchEmbedding])
    val heightPosGrid = stack(List.fill(widthExtent)(heightPosEncoded), newAxis = Axis[Width])

    concatenate(widthPosGrid, heightPosGrid, concatAxis = Axis[PatchEmbedding])

object VisitionTransformer2DPatching:

  case class Params[PatchWidth, PatchHeight, Channel, PatchEmbedding](
      projectionWeights: Tensor[(PatchWidth, PatchHeight, Channel, PatchEmbedding), Float]
  )

  object Params:

    def xavierUniform[PatchWidth: Label, PatchHeight: Label, Channel: Label, PatchEmbedding: Label](patchWidthExtent: AxisExtent[PatchWidth], patchHeightExtent: AxisExtent[PatchHeight], channelExtent: AxisExtent[Channel], embeddingExtent: AxisExtent[PatchEmbedding], key: Random.Key): Params[PatchWidth, PatchHeight, Channel, PatchEmbedding] =
      val flatProjectionWeights = init.xavierUniform(
        patchWidthExtent * patchHeightExtent * channelExtent,
        embeddingExtent,
        key
      )
      val projectionWeights = flatProjectionWeights.unflatten(
        Axis[PatchWidth |*| PatchHeight |*| Channel],
        Shape(patchWidthExtent, patchHeightExtent, channelExtent)
      )
      Params(projectionWeights = projectionWeights)
