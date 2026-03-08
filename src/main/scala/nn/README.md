# DeepWit

This document illustrates the core concepts behind DeepWit, a deep learning library built on top of DimWit.

## Design Philosophy

We share DimWit's vision: **put human understanding first.** 
In the context of a deep learning library, this means breaking away from the tradition of hiding concepts behind opaque, internal state or side effects.
Instead, DeepWit forces the user to be more explicit, to bring the code into tight alignment with the underlying theory.
This transparency also has a highly practical benefit: when the foundational concepts are explicit, building and understanding advanced techniques on top of them becomes significantly easier to reason about.

### The Cost of Hidden Concepts

Many libraries hide core concepts internally to make user code more compact, but at the cost of transparency.
To understand what the code is doing and link it back to theory, a reader must memorize framework-specific details. 
This forces the user to learn two disconnected things: the actual theoretical concepts, and the implementation specifics.

To illustrate this disconnect between theory and code, consider PyTorch's approach to automatic differentiation (backprop):
```python
# Setup: We have some code calculating the loss
loss = ...
# (1) Run backprop and set gradients internally in tensor objects.
loss.backward()
# (2) Apply gradients inside the tensor objects to the parameter values.
optimizer.step()
```
Without prior knowledge of the framework, it is impossible to deduce what happens under the hood at (1) and (2). Looking solely at the syntax, there is no indication of what is being optimized, or even that gradient-based learning is occurring.

The equivalent in DeepWit looks like this:
```scala
// Setup: We have a function defining how to calculate the loss
def lossF(...) = ...
// (1) Run automatic differentiation to calculate the gradients.
val grads = Autodiff.grad(lossF)(params)
// (2) Apply gradients to parameters to get new parameters.
val nextParams = optimizer.update(grads, params)
```
Because we must explicitly represent the gradients and the parameters, the code naturally aligns with the theory of gradient-based learning.

Furthermore, having explicit gradients makes implementing custom training logic—like gradient accumulation—trivial. Instead of relying on additional, framework-specific workarounds, the user simply sums the explicit gradient values over multiple batches.

### The Beauty of Explicit Concepts

In DeepWit, we embrace explicitness as a feature, not a burden. While defining concepts explicitly requires slightly more upfront code, the return on investment is massive: it demystifies complex architectures and tightly couples the implementation back to the theory.

Breaking with the tradition of most existing deep learning frameworks, DeepWit requires the user to, among others, represent learnable parameters as explicit data objects, pass hyperparameters in a separate parameter group, and handle random effects manually throughout the architecture (see Core Concepts in DeepWit).
These design choices replace "framework magic" with a transparent implementation that remains highly aligned with the theory. In this directness, we find the beauty.

## Core Concepts in DeepWit

### Explicit Parameter Representation

DeepWit requires parameters to be defined as dedicated data objects. While this adds a small amount of boilerplate, it brings the code into closer alignment with theory. It makes the implementation entirely transparent for critical tasks like weight initialization, parameter augmentation, and checkpointing.

For example, `AffineLayer.Params[In, Out]` represents the parameters for an affine layer. Because these are decoupled from the layer's logic, an explicit initialization method must be chosen at the time of creation. As the model's state is never hidden, tasks like checkpointing become as simple as storing the parameter data object.

```scala

object AffineLayer:

  case class Params[In, Out](
      weight: Tensor2[In, Out, Float],
      bias: Tensor1[Out, Float]
  )

  object Params:

    def xavierNormal[In: Label, Out: Label](
      inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key
    ): Params[In, Out] = Params(
      weight = init.xavierNormal(inExtent, outExtent, key),
      bias = Tensor(Shape(outExtent)).fill(0f)
    )

    def xavierUniform(...) = ...

```

### Explicit Hyperparameter group

DeepWit modules follow a curried constructor pattern that requires hyperparameters first, followed by parameters. This structural separation aligns the implementation with the theoretical transition from a model family to a mathematical function: Passing the hyperparameters fixes the "kind" of model, establishing the structure. Passing the parameters fixes the model's behavior to a concrete mathematical function.

```scala
case class Model
// 1: Define "kind" of model 
(hyperParams: HyperParams)
// 2: Define behavior of model
(params: Params)
// 3: Results in a concrete function
extends (In => Out):
  override apply(in: In): Out = ...
```

### Explicit Randomness

DeepWit treats stochasticity as an explicit capability rather than a hidden side effect, utilizing the Random Key concept (similar to JAX). This ensures that the presence—and absence—of randomness is clearly visible in the parameters of each module.

Most deep learning libraries—including PyTorch, TensorFlow, and even FLAX (built on top of JAX)—implement randomness in an implicit fashion by relying on effectful functions (e.g., torch.rand()) or global states (e.g., flax.nnx.Rngs). By breaking with this tradition, we ensure that a module's behavior is fully determined by its inputs. If a model is stochastic, its signature should say so.

## Downstream Benefits

Making things explicit has several, non-obvious benefits. Here are some highlights to showcase the benefits of this design philosophy.

### Mathematically Aligned Loss Functions

Having explicit parameters allows one to define a loss function as it is mathematically defined, as a function from parameters to a scalar loss.
Furthermore, we can define a more general loss function first taking the dataset (on which the loss in calculated) using Scala 3 function currying.

```scala
def loss(inputs: X, targets: Y)(
  params: Model.Params
): Tensor0[Float] = ...

val trainLoss = loss(trainX, trainY)
val valLoss = loss(valX, valY)
```
