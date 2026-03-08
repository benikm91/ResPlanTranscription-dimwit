# DeepWit

This document illustrates the concepts behind DeepWit, a deep learning library based on DimWit.

## Design Philosophie

We share DimWit's Vision to put human understanding first.
For deep learning, this means breaking with the tradition to make lots of concepts in code implicit, and force the user to explicitly define them - to make the code align with the theory.

### The Case against Implicit Concepts in Code

Implicit concepts in code means that code has opaque side effects not visible by looking at it. To understand the code and link it to theory, many implementation specific details must be known by a reader.

A straightforward examples is PyTorch's automatic differentiation and backprop implementation:
```python
loss = ... # Calculate loss
loss.backward()  # (1) Run reverse automatic differentiation (backprop) and set grad internally in the tensor objects.
optimizer.step() # (2) Apply gradients inside the tensor objects to the parameter values.
```
Note that the comments for (1) and (2) must be known beforehand to understand what is actually happening in these statements. There is no indication what is being optimized and that gradient-based learning takes place. 

The same in DeepWit would look like this:

```scala
def lossF(...) // Define loss function
val grads = Autodiff.grad(lossF)(params) // (1) calculate gradients for all params 
val nextParams = optimizer.update(grads, params) // apply gradients 
```
Note that the gradients are explicitly represented, making the code align with the theory of gradient-based algorithms.

In practice implicit (basic) concepts leads to more implict concepts. For example, in PyTorch gradient accumulation, a straightforward concept in theory, can't be clearly expressed as no explicit gradients exist in the first place! Therefore, PyTorch "implements" this by calling `loss.backwards()` multiple times; again this must be known by a coder. It is implicit and hard to understand what is going on.

### The case for Explicit Concepts in Code

In DeepWit we must express many concepts explicity. 
While we have some extra work to implement basics, it makes the link to theory and the implementation for more complex concepts easier. Overall, this leads to better understanding of the theory and code itself. Putting human understanding first.

Breaking with the tradition of most existing deep learning libraries, in DeepWit we must explicitly represent learnable parameters, hyper parameters, random effects, ... TODO.
In the following we discuss each in detail.

#### Explicit Parameter Representation

We require the user to define an explicit parameter representation.
While requiring extra code, it makes code closer to the deep learning theory. Especially, it makes implementations more transparent for concepts like weight initialization, parameter augmentation, checkpointing.

Here an example for explicit parameter representation: `AffineLayer.Params[In, Out]` represents the parameters for the `AffineLayer` module. The module further defines initialization method that clearly define how parameters are initialized. A user must explicitly choose (or implement) an initialization method when creating an affine layer as part of a bigger architecture.

```scala

object AffineLayer:

  case class Params[In, Out](
      weight: Tensor2[In, Out, Float],
      bias: Tensor1[Out, Float]
  )

```

#### Explicit Model

All modules in DeepWit follow the same basic design: a _curried_ constructor takes a first group of its hyper-parameter(s), followed by a group of its parameters. The module itself extends a function.
This aligns directly to the idea of a _model family_ and a _model_ from deep learning theory:
1. Model Family (Architecture): Defined by hyperparameters (e.g., number of layers, activation functions, etc.)
2. Concrete Model (Instance): Defined by trainable parameters (e.g., weights, biases, etc.)
3. An instance is a black-box, mathematical function.

```scala
case class Model(
    hyperParams: HyperParams // 1: Define model family (architecture)
)(
    params: Params           // 2: Define model (concrete model)
) extends (In => Out)        // 3: Providing parameters results in a function
```

#### Explicit Randomness

We require the user to define randomness explicitly through DimWit's key concept (i.e., JAX key concept). 
Most deep learning libraries like PyTorch, Tensorflow and even FLAX build on top of JAX, implement randomness in an implicit fashion by either relying on effectfull functions (e.g., `torch.rand()`) or global states (e.g., `flax.nnx.Rngs`). We break with this tradition.

If a architecture requires randomness, the user must explicitly take care of passing the keys for random effects to the layers and handle key splitting correctly inside the layers.
This makes randomness explicit, and forces the user to take care of. Most architecture don't use randomness and in DeepWit this lack of randomness indicates determinism (at least on a conceptual level, disregarding randomness due to the technical stack).
While dropout layer would add randomness during training to many common architectures, we implement dropout in DeepWit by parameter augmentation, making it part of the training algorithm not the architecture (see XXX).

### Examples for downstream benefits

Making things explicit has several, non-obvious benefits. Here are some highlights to showcase the benefits of this design philosophy.

#### Aligned Loss Functions

Having explicit parameters allows one to define a loss function as it is mathematically defined, as a function from parameters to a scalar loss.
Furthermore, we can define a more general loss function first taking the dataset (on which the loss in calculated) using Scala 3 function currying.

```scala
def loss(inputs: X, targets: Y)(
  params: Model.Params
): Tensor0[Float] = ...

val trainLoss = loss(trainX, trainY)
val valLoss = loss(valX, valY)
```
