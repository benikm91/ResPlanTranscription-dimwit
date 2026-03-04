package resplan.nn.transformer

import dimwit.*

def causalMask[Context: Label, CrossContext: Label](scoreShape: Shape2[Context, CrossContext]): Tensor[(Context, CrossContext), Boolean] =
  tril(noMask(scoreShape))

def noMask[Context: Label, CrossContext: Label](scoreShape: Shape2[Context, CrossContext]): Tensor[(Context, CrossContext), Boolean] =
  Tensor(scoreShape).fill(true)
