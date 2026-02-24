package resplan.nn

import dimwit.*

case class DimWitContext(key: Random.Key):
  def split(): (Random.Key, DimWitContext) =
    val (use, next) = key.split2()
    (use, DimWitContext(next))

// A context function that expects a DimWitContext
type DimWit[A] = DimWitContext ?=> A

def useKey[A](f: Random.Key => DimWit[A]): DimWit[A] =
  val (toUse, nextContext) = summon[DimWitContext].split()
  f(toUse)(using nextContext)
