package resplan.data

import scala.concurrent.Future
import scala.collection.mutable
import scala.concurrent.Await
import scala.concurrent.duration.*

object IteratorUtil:
  extension [T](it: Iterator[Future[T]])
    def awaitPrefetch(parallelism: Int, atMost: Duration = 10.minutes): Iterator[T] =
      val buffer = mutable.Queue.empty[Future[T]]

      def refill(): Unit =
        while buffer.size < parallelism && it.hasNext do
          buffer.enqueue(it.next())

      refill()
      new Iterator[T]:
        override def hasNext: Boolean = buffer.nonEmpty || it.hasNext

        override def next(): T =
          if buffer.isEmpty then throw new NoSuchElementException("next on empty iterator")
          val result = Await.result(buffer.dequeue(), atMost)
          refill()
          result
