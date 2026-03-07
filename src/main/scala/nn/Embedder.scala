package resplan.nn

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal
import dimwit.stats.Uniform

case class VocabularyEmbedder[Vocab: Label, Embedding: Label](params: VocabularyEmbedder.Params[Vocab, Embedding]) extends (Tensor0[Int] => Tensor1[Embedding, Float]):

  override def apply(token: Tensor0[Int]): Tensor1[Embedding, Float] =
    params.vocabularyEmbeddings.slice(Axis[Vocab].at(token))

object VocabularyEmbedder:

  case class Params[Vocab, Embedding](vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float])

  object Params:

    def lecunUniform[Vocab: Label, Embedding: Label](vocabExtent: AxisExtent[Vocab], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Vocab, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      val a = gain * (3f * variance).sqrt
      Params(IndependentDistribution.fromUnivariate(Shape(vocabExtent, embeddingExtent), Uniform(-a, a)).sample(key))

    def lecunNormal[Vocab: Label, Embedding: Label](vocabExtent: AxisExtent[Vocab], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Vocab, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      Params(Normal.standardIsotropic(Shape(vocabExtent, embeddingExtent), scale = gain * variance.sqrt).sample(key))

case class AddAbsolutePositionalEncoding[Context: Label, Embedding: Label](params: AddAbsolutePositionalEncoding.Params[Context, Embedding]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    context + params.positionalEmbeddings

object AddAbsolutePositionalEncoding:

  case class Params[Context, Embedding](positionalEmbeddings: Tensor2[Context, Embedding, Float])

  object Params:

    def lecunUniform[Context: Label, Embedding: Label](contextExtent: AxisExtent[Context], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Context, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      val a = gain * (3f * variance).sqrt
      Params(IndependentDistribution.fromUnivariate(Shape(contextExtent, embeddingExtent), Uniform(-a, a)).sample(key))

    def lecunNormal[Context: Label, Embedding: Label](contextExtent: AxisExtent[Context], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Context, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      Params(Normal.standardIsotropic(Shape(contextExtent, embeddingExtent), scale = gain * variance.sqrt).sample(key))
