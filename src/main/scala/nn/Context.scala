package resplan.nn

import dimwit.*

sealed trait NoContext
object NoContext:
  case object NoContextImpl extends NoContext
  given NoContext = NoContextImpl

case class RandomContext(key: Random.Key):
  type Randomness[A] = RandomContext ?=> A

  private def split(): (Random.Key, RandomContext) =
    val (use, next) = key.split2()
    (use, RandomContext(next))

  def useKey[A](f: Random.Key => Randomness[A]): Randomness[A] =
    val (toUse, nextContext) = summon[RandomContext].split()
    f(toUse)(using nextContext)
