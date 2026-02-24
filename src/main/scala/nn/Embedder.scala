package resplan.nn

import dimwit.*

case class VocabularyEmbedder[Vocab: Label, Embedding: Label](vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float]) extends (Tensor0[Int] => Tensor1[Embedding, Float]):

  override def apply(token: Tensor0[Int]): Tensor1[Embedding, Float] =
    vocabularyEmbeddings.slice(Axis[Vocab].at(token))

case class AbsolutePositionalEncoding[Context: Label, Embedding: Label](positionalEmbeddings: Tensor2[Context, Embedding, Float]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    context + positionalEmbeddings
