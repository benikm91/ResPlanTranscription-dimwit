package resplan.util

object IteratorUtil:

  extension [T](it: Iterator[T])
    def tapEvery(n: Int)(f: (T, Int) => Unit): Iterator[T] =
      it
        .zipWithIndex
        .tapEach: (t, id) =>
          if id % n == (n - 1) then f(t, id)
        .map(_._1)

    def nextAfter(n: Int): T =
      it.drop(n).next()
